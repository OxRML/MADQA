#!/usr/bin/env python3
"""
VS-Pages (Visually-Summarized Pages) for HEAVEN Stage 1.

EXACT COPY from: https://github.com/juyeonnn/HEAVEN/blob/main/indexing/vs-page/
- assemble.py
- utils.py
- DLA.py

VS-Pages extract TITLE REGIONS from documents and stack them vertically
to create compact visual summaries for efficient Stage 1 filtering.
"""

import cv2
import math
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


# =============================================================================
# UTILS - Copied from HEAVEN/indexing/vs-page/utils.py
# =============================================================================

def clean_name(fname: str) -> str:
    """Remove image extensions from filename."""
    return fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')


def grid_concat(images: List[np.ndarray], padding: int = 5) -> np.ndarray:
    """
    Create a grid layout for multiple images.
    Copied from: HEAVEN/indexing/vs-page/utils.py
    """
    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    num_images = len(images)
    grid_size = int(math.ceil(math.sqrt(num_images)))
    grid_rows = grid_size
    grid_cols = grid_size
    
    # Adjust grid to be more rectangular if needed
    if grid_size * (grid_size - 1) >= num_images:
        grid_rows = grid_size - 1
    
    # Calculate average aspect ratio
    aspect_ratios = [img.shape[1] / max(img.shape[0], 1) for img in images]
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
    
    # Calculate target cell size
    total_area = sum(img.shape[0] * img.shape[1] for img in images)
    avg_area = total_area / num_images
    target_height = int(math.sqrt(avg_area / max(avg_aspect, 0.1)))
    target_width = int(avg_aspect * target_height)
    target_height = max(target_height, 50)
    target_width = max(target_width, 50)
    
    # Resize all images
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        aspect = w / max(h, 1)
        
        if aspect > avg_aspect:
            new_width = target_width
            new_height = max(int(new_width / max(aspect, 0.1)), 1)
        else:
            new_height = target_height
            new_width = max(int(new_height * aspect), 1)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    
    # Create canvas
    canvas_width = grid_cols * target_width + (grid_cols - 1) * padding
    canvas_height = grid_rows * target_height + (grid_rows - 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place images on grid
    for idx, img in enumerate(resized_images):
        if idx >= num_images:
            break
        
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * (target_height + padding)
        x_start = col * (target_width + padding)
        
        h, w = img.shape[:2]
        y_offset = (target_height - h) // 2
        x_offset = (target_width - w) // 2
        
        canvas[y_start + y_offset:y_start + y_offset + h,
               x_start + x_offset:x_start + x_offset + w] = img
    
    return canvas


def naive_concat(images: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple images vertically or horizontally.
    Copied from: HEAVEN/indexing/vs-page/utils.py
    """
    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]
    total_width = sum(widths)
    total_height = sum(heights)
    
    # Concatenate vertically
    if total_width > total_height:
        min_width = min(widths)
        resized_imgs = [cv2.resize(img, (min_width, int(img.shape[0] * min_width / max(img.shape[1], 1)))) 
                        for img in images]
        concatenated = np.vstack(resized_imgs)
    # Concatenate horizontally
    else:
        min_height = min(heights)
        resized_imgs = [cv2.resize(img, (int(img.shape[1] * min_height / max(img.shape[0], 1)), min_height)) 
                        for img in images]
        concatenated = np.hstack(resized_imgs)
    
    return concatenated


# =============================================================================
# DLA - Copied from HEAVEN/indexing/vs-page/DLA.py
# =============================================================================

class DLA:
    """
    Document Layout Analysis class using DocLayout-YOLO model.
    Copied from: HEAVEN/indexing/vs-page/DLA.py
    
    Class labels (DocStructBench - 11 classes):
        0: Caption
        1: Footnote
        2: Formula
        3: List-item
        4: Page-footer
        5: Page-header
        6: Picture
        7: Section-header  <-- This is what we extract for VS-pages
        8: Table
        9: Text
        10: Title  <-- Also extract this
    """
    
    CLASS_NAMES = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title"
    }
    
    def __init__(self, device: str = "0"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load DocLayout-YOLO model - exact same as HEAVEN."""
        try:
            from huggingface_hub import hf_hub_download
            from doclayout_yolo import YOLOv10
            
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
            self.model = YOLOv10(model_path)
            print(f"DLA model loaded on device: {self.device}")
        except Exception as e:
            print(f"Warning: Could not load DocLayout-YOLO: {e}")
            print("VS-pages will use fallback grid method")
            self.model = None
    
    def get_layout(self, image: np.ndarray, imgsz: int = 1024, conf: float = 0.2) -> Dict:
        """
        Predict layout for a single image.
        Copied from: HEAVEN/indexing/vs-page/DLA.py get_layout()
        
        Args:
            image: Image as numpy array (BGR)
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary containing original shape and bounding boxes
        """
        if self.model is None:
            return {"orig_shape": image.shape[:2], "bbox": []}
        
        output = self.model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            device=f"cuda:{self.device}" if self.device != "cpu" else "cpu",
            verbose=False
        )
        output = output[0]
        
        ret = {"orig_shape": output.boxes.orig_shape, "bbox": []}
        for o in output:
            for cls, conf_val, xyxyn in zip(
                o.boxes.cls, o.boxes.conf, o.boxes.xyxyn
            ):
                ret["bbox"].append({
                    "cls": int(cls.cpu().tolist()),
                    "conf": conf_val.cpu().tolist(),
                    "xyxyn": xyxyn.cpu().tolist()
                })
        
        return ret


# =============================================================================
# ASSEMBLE - Copied from HEAVEN/indexing/vs-page/assemble.py
# =============================================================================

class Assemble:
    """
    Document Assembly class for creating VS-page images.
    Copied from: HEAVEN/indexing/vs-page/assemble.py
    
    Note: We use different CLASS_NAMES from DLA - this matches the original
    Assemble.py which uses DocStructBench schema:
        0: 'title'
        1: 'plain_text'
        ...
    
    But since DLA.py uses different schema (10: 'Title', 7: 'Section-header'),
    we need to handle both. The original HEAVEN uses class 0 ('title') but
    the actual DLA model outputs class 7 or 10 for headers/titles.
    """
    
    # Original Assemble.py CLASS_NAMES (DocStructBench alternate schema)
    CLASS_NAMES_ASSEMBLE = {
        0: 'title',
        1: 'plain_text',
        2: 'abandon',
        3: 'figure',
        4: 'figure_caption',
        5: 'table',
        6: 'table_caption',
        7: 'table_footnote',
        8: 'isolate_formula',
        9: 'formula_caption'
    }
    
    # DLA.py CLASS_NAMES (actual model output)
    CLASS_NAMES_DLA = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title"
    }
    
    # Target classes for VS-page extraction (title-related elements)
    # From DLA schema: 7 = Section-header, 10 = Title
    TARGET_CLASSES_DLA = [7, 10]
    
    def __init__(self, reduction_factor: int = 15, chunk_threshold: int = 20):
        """
        Initialize Assemble.
        
        Args:
            reduction_factor: Target reduction factor for pages
            chunk_threshold: Page threshold for creating chunks
        """
        self.reduction_factor = reduction_factor
        self.chunk_threshold = chunk_threshold
        self.dla = None
    
    def _ensure_dla(self, device: str = "0"):
        """Lazy load DLA model."""
        if self.dla is None:
            self.dla = DLA(device=device)
    
    def make_class_mask(
        self,
        page_bboxes: List[Dict],
        target_classes: List[int]
    ) -> List[bool]:
        """
        Create a boolean mask for bboxes based on their class.
        Copied from: HEAVEN/indexing/vs-page/assemble.py
        """
        cls = [b['cls'] for b in page_bboxes]
        cls_mask = [True if c in target_classes else False for c in cls]
        return cls_mask
    
    def get_title_regions(
        self,
        page_images: List[np.ndarray],
        bboxes: List[List[List[float]]],
        bbox_cls_masks: List[List[bool]]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract title regions from pages based on bounding boxes and class masks.
        Copied from: HEAVEN/indexing/vs-page/assemble.py get_title_regions()
        """
        title_regions = []
        page_sources = []
        
        for page_num, (img, page_bboxes, bbox_cls_mask) in enumerate(
            zip(page_images, bboxes, bbox_cls_masks)
        ):
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            for bbox, cls_mask in zip(page_bboxes, bbox_cls_mask):
                if not cls_mask:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Ensure valid region
                if x2 <= x1 or y2 <= y1:
                    continue
                
                region = img[y1:y2, x1:x2]
                
                # Rotate tall narrow regions (likely rotated text)
                if region.shape[0] > region.shape[1] * 5:
                    region = np.rot90(region, k=-1)
                
                if region.size > 0:
                    title_regions.append(region)
                    page_sources.append(page_num)
        
        return title_regions, page_sources
    
    def gen_vs_page(self, title_regions: List[np.ndarray]) -> np.ndarray:
        """
        Generate a vs-page image from title regions using padding method.
        Stacks regions vertically with centered alignment.
        Copied from: HEAVEN/indexing/vs-page/assemble.py gen_vs_page()
        """
        if not title_regions:
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Stack vertically with centered alignment
        max_width = max(region.shape[1] for region in title_regions)
        padded_regions = []
        
        for region in title_regions:
            # Create white canvas with max width
            padded = np.ones((region.shape[0], max_width, 3), dtype=np.uint8) * 255
            # Center the region horizontally
            x_offset = (max_width - region.shape[1]) // 2
            padded[:, x_offset:x_offset + region.shape[1]] = region
            padded_regions.append(padded)
        
        # Stack all regions vertically
        vs_page = np.vstack(padded_regions)
        return vs_page
    
    def construct_vs_page(
        self,
        page_images: List[np.ndarray],
        bboxes: List[List[List[float]]],
        bbox_cls_masks: List[List[bool]],
        region_min_threshold: int = 4,
        page_len: int = 0
    ) -> Tuple[List[np.ndarray], Dict[int, List[int]]]:
        """
        Extract title regions and create vs-page images with chunking.
        Copied from: HEAVEN/indexing/vs-page/assemble.py construct_vs_page()
        
        Args:
            page_images: List of page images as numpy arrays
            bboxes: List of normalized bounding boxes
            bbox_cls_masks: List of boolean masks for filtering bboxes
            region_min_threshold: Minimum number of regions to create chunks
            page_len: Total number of pages in document
            
        Returns:
            Tuple of (vs_page_images, chunk_to_pages_mapping)
        """
        vs_pages = []
        chunk_mapping = {}  # chunk_idx -> list of page indices
        
        title_regions, page_sources = self.get_title_regions(page_images, bboxes, bbox_cls_masks)
        
        if page_len == 0:
            page_len = len(page_images)
        
        # Case 1: If too few regions, concatenate entire pages
        if len(title_regions) < region_min_threshold:
            if len(page_images) > 3:
                vs_image = grid_concat(page_images)
            else:
                vs_image = naive_concat(page_images)
            vs_pages.append(vs_image)
            chunk_mapping[0] = list(range(len(page_images)))
            return vs_pages, chunk_mapping
        
        # Case 2: If document is small, create single vs-page
        if page_len <= self.chunk_threshold:
            vs_image = self.gen_vs_page(title_regions)
            vs_pages.append(vs_image)
            chunk_mapping[0] = sorted(list(set(page_sources)))
            return vs_pages, chunk_mapping
        
        # Case 3: Create multiple chunks
        region_per_page = math.ceil(len(title_regions) * (self.reduction_factor / page_len))
        num_chunks = len(title_regions) // region_per_page
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * region_per_page
            end_idx = start_idx + region_per_page
            
            title_regions_chunk = title_regions[start_idx:end_idx]
            source_pages = set(page_sources[start_idx:end_idx])
            
            vs_image = self.gen_vs_page(title_regions_chunk)
            vs_pages.append(vs_image)
            chunk_mapping[chunk_idx] = sorted(list(source_pages))
        
        # Handle remaining regions
        if len(title_regions) % region_per_page != 0:
            start_idx = num_chunks * region_per_page
            end_idx = len(title_regions)
            
            title_regions_chunk = title_regions[start_idx:end_idx]
            source_pages = set(page_sources[start_idx:end_idx])
            
            vs_image = self.gen_vs_page(title_regions_chunk)
            vs_pages.append(vs_image)
            chunk_mapping[num_chunks] = sorted(list(source_pages))
        
        return vs_pages, chunk_mapping
    
    def postprocess_mapping(
        self,
        chunk_mapping: Dict[int, List[int]],
        num_pages: int
    ) -> Dict[int, List[int]]:
        """
        Post-process chunk mappings to ensure all pages are included.
        Copied from: HEAVEN/indexing/vs-page/assemble.py postprocess()
        
        This fills in any gaps in the chunk mappings by distributing pages
        into the chunks based on their order.
        """
        if not chunk_mapping:
            return {0: list(range(num_pages))}
        
        # Sort chunks by their first page
        sorted_chunks = sorted(chunk_mapping.items(), key=lambda x: min(x[1]) if x[1] else 0)
        
        # Create new mapping with all pages distributed
        new_mapping = {}
        all_assigned = set()
        
        for chunk_idx, pages in sorted_chunks:
            all_assigned.update(pages)
        
        # Find unassigned pages
        unassigned = set(range(num_pages)) - all_assigned
        
        if not unassigned:
            return chunk_mapping
        
        # Distribute unassigned pages to nearest chunks
        for page_idx in sorted(unassigned):
            # Find the best chunk for this page
            best_chunk = 0
            best_distance = float('inf')
            
            for chunk_idx, pages in chunk_mapping.items():
                if pages:
                    # Distance to this chunk's page range
                    min_page = min(pages)
                    max_page = max(pages)
                    
                    if min_page <= page_idx <= max_page:
                        # Page falls within this chunk's range
                        best_chunk = chunk_idx
                        break
                    else:
                        distance = min(abs(page_idx - min_page), abs(page_idx - max_page))
                        if distance < best_distance:
                            best_distance = distance
                            best_chunk = chunk_idx
            
            chunk_mapping[best_chunk].append(page_idx)
        
        # Sort pages within each chunk
        for chunk_idx in chunk_mapping:
            chunk_mapping[chunk_idx] = sorted(list(set(chunk_mapping[chunk_idx])))
        
        return chunk_mapping


# =============================================================================
# BUILD VS-PAGE INDEX - Main entry point
# =============================================================================

def build_vs_page_index(
    images: List[Image.Image],
    stage1_encoder,
    reduction_factor: int = 15,
    use_dla: bool = True,
    device: str = "0",
    batch_size: int = 16
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Build VS-page index for Stage 1 retrieval.
    Matches HEAVEN paper implementation exactly.
    
    Args:
        images: List of document page images (PIL)
        stage1_encoder: Single-vector encoder (e.g., DSE)
        reduction_factor: Target reduction factor for pages
        use_dla: Whether to use Document Layout Analysis
        device: CUDA device
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of:
        - vs_page_embeddings: Embeddings for VS-pages
        - vs_page_metadata: Metadata mapping VS-pages to source pages  
        - page_embeddings: Embeddings for individual pages
    """
    print(f"Building VS-page index (reduction_factor={reduction_factor}, use_dla={use_dla})")
    
    # Convert PIL images to numpy arrays (BGR for cv2)
    page_arrays = []
    for img in images:
        arr = np.array(img.convert('RGB'))  # Ensure RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert to BGR for cv2
        page_arrays.append(arr)
    
    assembler = Assemble(reduction_factor=reduction_factor)
    
    if use_dla:
        # Run DLA on all pages
        print("Running Document Layout Analysis...")
        assembler._ensure_dla(device)
        
        layouts = []
        for img in tqdm(page_arrays, desc="Analyzing layouts (DLA)"):
            layout = assembler.dla.get_layout(img)
            layouts.append(layout)
        
        # Extract bboxes and create masks
        bboxes = []
        bbox_cls_masks = []
        
        for layout in layouts:
            page_bboxes = layout.get('bbox', [])
            # Extract xyxyn coordinates
            page_bbox_coords = [b['xyxyn'] for b in page_bboxes]
            bboxes.append(page_bbox_coords)
            
            # Create class mask for title elements
            # Using DLA schema: 7 = Section-header, 10 = Title
            cls_mask = assembler.make_class_mask(page_bboxes, Assemble.TARGET_CLASSES_DLA)
            bbox_cls_masks.append(cls_mask)
        
        # Construct VS-pages from title regions
        print("Constructing VS-pages from title regions...")
        vs_page_arrays, chunk_mapping = assembler.construct_vs_page(
            page_images=page_arrays,
            bboxes=bboxes,
            bbox_cls_masks=bbox_cls_masks,
            page_len=len(page_arrays)
        )
        
        # Post-process to ensure all pages are mapped
        chunk_mapping = assembler.postprocess_mapping(chunk_mapping, len(page_arrays))
        
    else:
        # Simple fallback: group pages into chunks and use grid layout
        print("Creating grid-based VS-pages (no DLA)...")
        num_pages = len(page_arrays)
        num_vs_pages = max(1, (num_pages + reduction_factor - 1) // reduction_factor)
        
        vs_page_arrays = []
        chunk_mapping = {}
        
        for vs_idx in tqdm(range(num_vs_pages), desc="Creating VS-pages"):
            start_idx = vs_idx * reduction_factor
            end_idx = min(start_idx + reduction_factor, num_pages)
            page_range = list(range(start_idx, end_idx))
            
            # Create grid of page images
            chunk_images = [page_arrays[i] for i in page_range]
            vs_image = grid_concat(chunk_images)
            
            vs_page_arrays.append(vs_image)
            chunk_mapping[vs_idx] = page_range
    
    # Convert VS-page numpy arrays back to PIL for encoding
    vs_pages_pil = []
    for arr in vs_page_arrays:
        # Convert BGR back to RGB
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        vs_pages_pil.append(Image.fromarray(arr_rgb))
    
    print(f"Created {len(vs_pages_pil)} VS-pages from {len(images)} pages")
    
    # Build metadata
    vs_metadata = []
    for chunk_idx, page_range in sorted(chunk_mapping.items()):
        vs_metadata.append({
            "vs_page_idx": chunk_idx,
            "page_range": page_range,
            "method": "dla_title_extraction" if use_dla else "grid_fallback"
        })
    
    # Encode VS-pages
    print("Encoding VS-pages...")
    vs_page_embeddings = stage1_encoder.encode_images(vs_pages_pil, batch_size=batch_size)
    
    # Encode individual pages
    print("Encoding individual pages...")
    page_embeddings = stage1_encoder.encode_images(images, batch_size=batch_size)
    
    return vs_page_embeddings, vs_metadata, page_embeddings
