#!/usr/bin/env python3
"""
Document Layout Analysis (DLA) for HEAVEN.

Copied directly from: https://github.com/juyeonnn/HEAVEN/blob/main/indexing/vs-page/DLA.py

Uses DocLayout-YOLO model for layout detection.
"""

from huggingface_hub import hf_hub_download
import cv2
import os 
import json
from tqdm import tqdm
import math
from typing import Dict, Optional, List

from heaven_utils import filter_files, clean_name


class DLA:
    """Document Layout Analysis class using DocLayout-YOLO model."""
    
    # Class labels mapping
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
    
    def __init__(self, device: str = "0", model_path: Optional[str] = None):
        """
        Initialize the DLA model.
        
        Args:
            device: CUDA device ID (e.g., "0", "1") or "cpu"
            model_path: Optional path to model file. If None, downloads from HuggingFace.
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        
    def _load_model(self):
        """Load the DocLayout-YOLO model."""
        if self.model is not None:
            return
            
        try:
            from doclayout_yolo import YOLOv10
            
            if self.model_path is None:
                self.model_path = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
                )
            self.model = YOLOv10(self.model_path)
            print(f"DLA model loaded on device: {self.device}")
        except ImportError:
            print("Warning: doclayout_yolo not installed")
            print("Install with: pip install doclayout-yolo")
            self.model = None

    def get_layout(self, image_path: str, imgsz: int = 1024, conf: float = 0.2) -> Dict:
        """
        Predict layout for a single image.
        
        Args:
            image_path: Path to the image file
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary containing original shape and bounding boxes with layout information
        """
        self._load_model()
        
        if self.model is None:
            return {"orig_shape": None, "bbox": []}
        
        output = self.model.predict(
            image_path,
            imgsz=imgsz,
            conf=conf,
            device=f"cuda:{self.device}",
            verbose=False
        )
        output = output[0]
        
        ret = {"orig_shape": output.boxes.orig_shape, "bbox": []}
        for o in output:
            for cls, conf_score, xywh, xywhn, xyxy, xyxyn in zip(
                o.boxes.cls, o.boxes.conf, o.boxes.xywh, 
                o.boxes.xywhn, o.boxes.xyxy, o.boxes.xyxyn
            ):
                ret["bbox"].append({
                    "cls": cls.cpu().tolist(),
                    "conf": conf_score.cpu().tolist(),
                    "xyxyn": xyxyn.cpu().tolist()
                })
        
        return ret
    
    def process_images(
        self, 
        image_paths: List[str],
        imgsz: int = 1024,
        conf: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Process multiple images and return layout predictions.
        
        Args:
            image_paths: List of paths to image files
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary mapping image names to layout predictions
        """
        self._load_model()
        
        data = {}
        for image_path in tqdm(image_paths, desc="Analyzing layouts"):
            fname = os.path.basename(image_path)
            key = clean_name(fname)
            try:
                ret = self.get_layout(image_path, imgsz=imgsz, conf=conf)
                data[key] = ret
            except Exception as e:
                print(f"Error in {fname}: {str(e)}")    
                data[key] = {"orig_shape": None, "bbox": []}
        
        return data
    
    def process_dataset(
        self, 
        dataset_path: str, 
        pages_dir: str = "pages",
        output_json: Optional[str] = None,
        imgsz: int = 1024,
        conf: float = 0.2
    ) -> Dict:
        """
        Process all images in a dataset directory.
        
        Args:
            dataset_path: Path to the dataset directory
            pages_dir: Subdirectory containing page images
            output_json: Path to save JSON results. If None, uses default location.
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary with all layout predictions
        """
        page_dir = os.path.join(dataset_path, pages_dir)
        
        if not os.path.exists(page_dir):
            raise ValueError(f"Page directory not found: {page_dir}")
        
        if output_json is None:
            output_json = os.path.join(dataset_path, "layout.json")
        
        # Get all image files
        files = os.listdir(page_dir)
        files = filter_files(files)
        image_paths = [os.path.join(page_dir, f) for f in files]
        
        print(f"Processing {len(image_paths)} images from {page_dir}")
        
        # Process images
        data = self.process_images(image_paths, imgsz=imgsz, conf=conf)
        
        # Save results
        with open(output_json, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"Saved layout to {output_json}")

        return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Layout Analysis")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Path to dataset directory")
    parser.add_argument("--device", type=str, default="0",
                       help="CUDA device ID or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=1024,
                       help="Prediction image size")
    parser.add_argument("--conf", type=float, default=0.2,
                       help="Confidence threshold")
    args = parser.parse_args()
    
    # Initialize and run DLA
    dla = DLA(device=args.device)
    dla.process_dataset(
        dataset_path=args.dataset,
        imgsz=args.imgsz,
        conf=args.conf
    )
    print("\n=== DLA Complete ===")

