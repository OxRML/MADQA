#!/usr/bin/env python3
"""
Document Indexing for HEAVEN Retrieval.

Based on: https://github.com/juyeonnn/HEAVEN
Paper: https://arxiv.org/pdf/2510.22215

Handles:
1. PDF to image conversion
2. VS-Page (Visually-Summarized Pages) construction
3. Document embedding generation  
4. Index storage and loading
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
from tqdm import tqdm


def pdf_to_images(pdf_path: Path, dpi: int = 144, max_pixels: int = 4000000) -> List[Image.Image]:
    """
    Convert PDF to list of PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering (144 is good balance of quality/size)
        max_pixels: Maximum pixels per page (4MP default). Larger pages are scaled down.
    
    Returns:
        List of PIL Images, one per page
    """
    import signal
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Please install PyMuPDF: pip install pymupdf")
    
    # Adjust DPI for large files (>20MB)
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:
        dpi = 72  # Low quality for very large files
    elif file_size_mb > 20:
        dpi = 100  # Medium quality for large files
    
    doc = fitz.open(pdf_path)
    images = []
    
    # Timeout handler
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Page rendering timed out")
    
    for page_num in range(len(doc)):
        try:
            # Set 30 second timeout per page
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            page = doc[page_num]
            
            # Calculate scale to limit max pixels
            rect = page.rect
            page_pixels = (rect.width * dpi / 72) * (rect.height * dpi / 72)
            
            if page_pixels > max_pixels:
                # Scale down to max_pixels
                scale = (max_pixels / page_pixels) ** 0.5
                effective_dpi = int(dpi * scale)
            else:
                effective_dpi = dpi
            
            # Render page to image
            mat = fitz.Matrix(effective_dpi / 72, effective_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
            # Cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
        except TimeoutError:
            print(f"\n  Timeout on page {page_num} of {pdf_path.name}, using blank page")
            signal.alarm(0)
            # Create a blank page as placeholder
            images.append(Image.new("RGB", (595, 842), color=(255, 255, 255)))
        except Exception as e:
            print(f"\n  Error on page {page_num} of {pdf_path.name}: {e}")
            images.append(Image.new("RGB", (595, 842), color=(255, 255, 255)))
    
    doc.close()
    return images


class DocumentIndex:
    """
    Document index for HEAVEN two-stage retrieval.
    
    Stores:
    - VS-Page embeddings (for efficient Stage 1 filtering)
    - Stage 1 single-vector embeddings (for page-level retrieval)
    - Stage 2 multi-vector embeddings (for accurate re-ranking)
    - VS-Page to page mapping
    - Document metadata (PDF names, page numbers)
    """
    
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Document metadata: list of (pdf_name, page_num) tuples
        self.doc_metadata: List[Tuple[str, int]] = []
        
        # VS-Page components
        self.vs_page_embeddings: Optional[np.ndarray] = None  # VS-page level embeddings
        self.vs_page_metadata: Optional[List[Dict]] = None  # VS-page to page mapping
        
        # Stage 1: Single-vector embeddings (N x D numpy array)
        self.stage1_embeddings: Optional[np.ndarray] = None
        
        # Stage 2: Multi-vector embeddings (list of numpy arrays)
        self.stage2_embeddings: Optional[List[np.ndarray]] = None
        
        # Document images (optional, for VLM-based QA)
        self.doc_images: Optional[List[Image.Image]] = None
        
        # Index configuration
        self.config: Dict[str, Any] = {}
        
    def build_index(
        self,
        pdf_dir: Path,
        stage1_encoder=None,
        stage2_encoder=None,
        limit: Optional[int] = None,
        batch_size: int = 16,  # Default for large VRAM (140GB+)
        save_images: bool = True,
        # VS-Page parameters
        use_vs_pages: bool = True,
        reduction_factor: int = 15,
        use_dla: bool = False,  # DLA is expensive, default to simple mode
        # Skip options for faster re-indexing
        skip_pdf_conversion: bool = False  # Load from saved images instead
    ):
        """
        Build index from directory of PDFs.
        
        Args:
            pdf_dir: Directory containing PDF files
            stage1_encoder: Encoder for stage 1 (single-vector)
            stage2_encoder: Encoder for stage 2 (multi-vector), optional
            limit: Max number of PDFs to process
            batch_size: Batch size for encoding
            save_images: Whether to save document images
            use_vs_pages: Whether to build VS-page index for efficient retrieval
            reduction_factor: Number of pages per VS-page
            use_dla: Whether to use Document Layout Analysis for VS-pages
            skip_pdf_conversion: Skip PDF conversion, load from saved images instead
        """
        all_images = []
        self.doc_metadata = []
        
        # Option 1: Load from previously saved images
        if skip_pdf_conversion:
            images_dir = self.index_dir / "index_images"
            metadata_path = self.index_dir / "index_metadata.json"
            
            if images_dir.exists() and metadata_path.exists():
                print(f"Loading images from {images_dir}...")
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.doc_metadata = [tuple(x) for x in metadata["doc_metadata"]]
                
                image_files = sorted(images_dir.glob("*.jpg"))
                if limit:
                    image_files = image_files[:limit]
                    self.doc_metadata = self.doc_metadata[:limit]
                
                for img_path in tqdm(image_files, desc="Loading images"):
                    all_images.append(Image.open(img_path))
                
                print(f"Loaded {len(all_images)} images from cache")
            else:
                print(f"No cached images found at {images_dir}, falling back to PDF conversion")
                skip_pdf_conversion = False
        
        # Option 2: Convert PDFs to images
        if not skip_pdf_conversion:
            pdf_dir = Path(pdf_dir)
            pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
            
            if limit:
                pdf_files = pdf_files[:limit]
            
            print(f"Processing {len(pdf_files)} PDFs from {pdf_dir}")
            
            # Convert PDFs to images
            for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
                try:
                    images = pdf_to_images(pdf_path)
                    for page_num, img in enumerate(images):
                        all_images.append(img)
                        self.doc_metadata.append((pdf_path.name, page_num))
                except Exception as e:
                    print(f"Error processing {pdf_path.name}: {e}")
        
        print(f"Total pages: {len(all_images)}")
        
        # Store images if requested
        if save_images:
            self.doc_images = all_images
        
        # Store configuration
        # Count unique PDFs from metadata
        num_pdfs = len(set(m[0] for m in self.doc_metadata)) if self.doc_metadata else 0
        self.config = {
            "use_vs_pages": use_vs_pages,
            "reduction_factor": reduction_factor,
            "use_dla": use_dla,
            "num_pages": len(all_images),
            "num_pdfs": num_pdfs
        }
        
        # Build VS-Page index if enabled
        if use_vs_pages and stage1_encoder is not None:
            print(f"\nBuilding VS-Page index (reduction_factor={reduction_factor})...")
            from vs_pages import build_vs_page_index
            
            vs_emb, vs_meta, page_emb = build_vs_page_index(
                images=all_images,
                stage1_encoder=stage1_encoder,
                reduction_factor=reduction_factor,
                use_dla=use_dla,
                device=stage1_encoder.device,
                batch_size=batch_size
            )
            
            self.vs_page_embeddings = vs_emb
            self.vs_page_metadata = vs_meta
            self.stage1_embeddings = page_emb
            
            print(f"VS-Page embeddings shape: {vs_emb.shape}")
            print(f"Page embeddings shape: {page_emb.shape}")
            
        elif stage1_encoder is not None:
            # Build standard Stage 1 embeddings without VS-pages
            print("Building Stage 1 (single-vector) embeddings...")
            self.stage1_embeddings = stage1_encoder.encode_images(
                all_images, 
                batch_size=batch_size
            )
            print(f"Stage 1 embeddings shape: {self.stage1_embeddings.shape}")
        
        # Build Stage 2 embeddings (optional, more expensive)
        if stage2_encoder is not None:
            print("Building Stage 2 (multi-vector) embeddings...")
            self.stage2_embeddings = stage2_encoder.encode_images(
                all_images,
                batch_size=batch_size
            )
            print(f"Stage 2 embeddings: {len(self.stage2_embeddings)} documents")
    
    def save(self, name: str = "index"):
        """Save index to disk."""
        # Save metadata
        metadata_path = self.index_dir / f"{name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "doc_metadata": self.doc_metadata,
                "num_docs": len(self.doc_metadata),
                "has_vs_pages": self.vs_page_embeddings is not None,
                "has_stage1": self.stage1_embeddings is not None,
                "has_stage2": self.stage2_embeddings is not None,
                "has_images": self.doc_images is not None,
                "config": self.config
            }, f, indent=2)
        
        # Save VS-page embeddings and metadata
        if self.vs_page_embeddings is not None:
            vs_emb_path = self.index_dir / f"{name}_vs_pages.npy"
            np.save(vs_emb_path, self.vs_page_embeddings)
            
            vs_meta_path = self.index_dir / f"{name}_vs_pages_meta.json"
            with open(vs_meta_path, "w") as f:
                json.dump(self.vs_page_metadata, f, indent=2)
        
        # Save Stage 1 embeddings
        if self.stage1_embeddings is not None:
            stage1_path = self.index_dir / f"{name}_stage1.npy"
            np.save(stage1_path, self.stage1_embeddings)
        
        # Save Stage 2 embeddings
        if self.stage2_embeddings is not None:
            stage2_path = self.index_dir / f"{name}_stage2.pkl"
            with open(stage2_path, "wb") as f:
                pickle.dump(self.stage2_embeddings, f)
        
        # Save images (compressed)
        if self.doc_images is not None:
            images_dir = self.index_dir / f"{name}_images"
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(tqdm(self.doc_images, desc="Saving images")):
                img.save(images_dir / f"{i:06d}.jpg", quality=85)
        
        print(f"Index saved to {self.index_dir}")
    
    def load(self, name: str = "index"):
        """Load index from disk."""
        # Load metadata
        metadata_path = self.index_dir / f"{name}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.doc_metadata = [tuple(x) for x in metadata["doc_metadata"]]
        self.config = metadata.get("config", {})
        
        # Load VS-page embeddings
        vs_emb_path = self.index_dir / f"{name}_vs_pages.npy"
        if vs_emb_path.exists():
            self.vs_page_embeddings = np.load(vs_emb_path)
            print(f"Loaded VS-page embeddings: {self.vs_page_embeddings.shape}")
            
            vs_meta_path = self.index_dir / f"{name}_vs_pages_meta.json"
            if vs_meta_path.exists():
                with open(vs_meta_path, "r") as f:
                    self.vs_page_metadata = json.load(f)
        
        # Load Stage 1 embeddings
        stage1_path = self.index_dir / f"{name}_stage1.npy"
        if stage1_path.exists():
            self.stage1_embeddings = np.load(stage1_path)
            print(f"Loaded Stage 1 embeddings: {self.stage1_embeddings.shape}")
        
        # Load Stage 2 embeddings
        stage2_path = self.index_dir / f"{name}_stage2.pkl"
        if stage2_path.exists():
            with open(stage2_path, "rb") as f:
                self.stage2_embeddings = pickle.load(f)
            print(f"Loaded Stage 2 embeddings: {len(self.stage2_embeddings)} documents")
        
        # Load images
        images_dir = self.index_dir / f"{name}_images"
        if images_dir.exists():
            image_files = sorted(images_dir.glob("*.jpg"))
            self.doc_images = [Image.open(f) for f in image_files]
            print(f"Loaded {len(self.doc_images)} document images")
        
        print(f"Index loaded: {len(self.doc_metadata)} documents")
        if self.config:
            print(f"Config: VS-pages={self.config.get('use_vs_pages')}, "
                  f"reduction_factor={self.config.get('reduction_factor')}")
    
    def get_document_info(self, idx: int) -> Dict[str, Any]:
        """Get document info by index."""
        pdf_name, page_num = self.doc_metadata[idx]
        return {
            "index": idx,
            "pdf_name": pdf_name,
            "page_num": page_num,
            "image": self.doc_images[idx] if self.doc_images else None
        }
    
    def get_pages_for_vs_page(self, vs_page_idx: int) -> List[int]:
        """Get page indices that belong to a VS-page."""
        if self.vs_page_metadata is None:
            return []
        
        if vs_page_idx >= len(self.vs_page_metadata):
            return []
        
        return self.vs_page_metadata[vs_page_idx].get("page_range", [])
    
    def get_vs_pages_for_page(self, page_idx: int) -> List[int]:
        """Get VS-page indices that contain a given page."""
        if self.vs_page_metadata is None:
            return []
        
        # Build reverse mapping if not cached
        if not hasattr(self, '_page_to_vs_pages') or self._page_to_vs_pages is None:
            self._page_to_vs_pages = {}
            for vs_idx, vs_meta in enumerate(self.vs_page_metadata):
                for p_idx in vs_meta.get("page_range", []):
                    if p_idx not in self._page_to_vs_pages:
                        self._page_to_vs_pages[p_idx] = []
                    self._page_to_vs_pages[p_idx].append(vs_idx)
        
        return self._page_to_vs_pages.get(page_idx, [])
