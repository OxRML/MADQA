#!/usr/bin/env python3
"""
Visual Document Encoders for HEAVEN Retrieval.

Directly copied from: https://github.com/juyeonnn/HEAVEN
- DSE encoder: indexing/encode/dse_encoder.py
- ColQwen2.5 encoder: indexing/encode/colqwen25_encoder.py

Stage 1 uses single-vector encoders for efficient initial retrieval.
Stage 2 uses multi-vector encoders for accurate re-ranking.
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
from tqdm import tqdm

# Allow loading truncated images (from original HEAVEN)
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseEncoder(ABC):
    """Abstract base class for document encoders."""
    
    def __init__(self, device: str = "0"):
        """
        Initialize encoder.
        
        Args:
            device: CUDA device ID (e.g., "0", "1") - matches original HEAVEN convention
        """
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load the specific model and processor."""
        pass
    
    @abstractmethod
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode document images."""
        pass
    
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode text queries."""
        pass


# =============================================================================
# DSE Encoder - Copied from HEAVEN/indexing/encode/dse_encoder.py
# =============================================================================

class DSEEncoder(BaseEncoder):
    """
    DSE encoder implementation.
    
    Copied from: https://github.com/juyeonnn/HEAVEN/blob/main/indexing/encode/dse_encoder.py
    """
    
    def __init__(self, device: str = "0", model_name: str = "MrLight/dse-qwen2-2b-mrl-v1"):
        self.model_name = model_name
        super().__init__(device)
    
    def _load_model(self, **kwargs):
        """Load DSE model and processor.
        
        Uses manual loading to avoid from_pretrained hanging issues,
        but ensures bfloat16 dtype for correct embeddings.
        """
        from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
        from huggingface_hub import hf_hub_download
        
        min_pixels = 1*28*28
        max_pixels = 2560*28*28
        device = f"cuda:{self.device}"
        
        print(f"Loading DSE model: {self.model_name}", flush=True)
        
        # Load processor
        print("  [1/5] Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        
        # Load config
        print("  [2/5] Loading config...", flush=True)
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Create model from config with bfloat16 (critical for correct embeddings!)
        print("  [3/5] Creating model (bfloat16)...", flush=True)
        self.model = Qwen2VLForConditionalGeneration(config).to(torch.bfloat16)
        
        # Load state dict
        print("  [4/5] Loading weights...", flush=True)
        try:
            from safetensors.torch import load_file
            path = hf_hub_download(self.model_name, 'model.safetensors')
            state_dict = load_file(path)
        except:
            path = hf_hub_download(self.model_name, 'pytorch_model.bin')
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        # Fix key prefix mismatch between state_dict and model:
        # - state_dict 'model.xxx' -> model expects 'model.language_model.xxx'
        # - state_dict 'visual.xxx' -> model expects 'model.visual.xxx'  
        # - state_dict 'lm_head.xxx' -> model expects 'lm_head.xxx' (no change)
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # model.layers -> model.language_model.layers
                new_key = key.replace('model.', 'model.language_model.', 1)
            elif key.startswith('visual.'):
                # visual.xxx -> model.visual.xxx
                new_key = 'model.' + key
            else:
                # lm_head.xxx stays the same
                new_key = key
            fixed_state_dict[new_key] = value
        
        result = self.model.load_state_dict(fixed_state_dict, strict=False)
        if result.missing_keys:
            print(f"  Warning: {len(result.missing_keys)} missing keys")
        if result.unexpected_keys:
            print(f"  Warning: {len(result.unexpected_keys)} unexpected keys")
        del state_dict, fixed_state_dict
        
        # Move to GPU
        print("  [5/5] Moving to GPU...", flush=True)
        self.model = self.model.to(device).eval()
        self.model.padding_side = "left"
        
        print("  DSE model loaded!", flush=True)
    
    def _get_embedding(self, last_hidden_state: torch.Tensor, dimension: int = 1536) -> torch.Tensor:
        """Extract embedding from hidden states.
        
        DSE uses MRL (Matryoshka Representation Learning) - embeddings must be
        sliced to the target dimension and then normalized.
        
        From DSE documentation:
        https://huggingface.co/MrLight/dse-qwen2-2b-mrl-v1
        """
        reps = last_hidden_state[:, -1]
        # MRL requires slicing to target dimension BEFORE normalizing
        reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
        return reps
    
    def _process_single_item(self, image: Image.Image) -> Tuple[torch.Tensor, None]:
        """Process a single image and return embedding."""
        from qwen_vl_utils import process_vision_info
        
        message = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': 'What is shown in this image?'}
                ]
            }
        ]

        doc_texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
        ]
        doc_image_inputs, doc_video_inputs = process_vision_info([message])
        doc_inputs = self.processor(
            text=doc_texts, 
            images=doc_image_inputs, 
            videos=doc_video_inputs, 
            padding='longest', 
            return_tensors='pt'
        ).to(f"cuda:{self.device}")
        
        cache_position = torch.arange(0, len(doc_texts))
        doc_inputs = self.model.prepare_inputs_for_generation(**doc_inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        
        # _get_embedding now handles MRL slicing and normalization
        doc_embeddings = self._get_embedding(output.hidden_states[-1], 1536)
        
        return doc_embeddings, None

    def _process_single_query(self, query: str) -> Tuple[torch.Tensor, None]:
        """Process a single query and return embedding."""
        from qwen_vl_utils import process_vision_info
        
        # Create dummy image for text-only processing
        message = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height':1, 'resized_width':1},
                    {'type': 'text', 'text': f'Query: {query}'},
                ]
            }
        ]

        query_texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"]
        query_image_inputs, query_video_inputs = process_vision_info(message)
        query_inputs = self.processor(
            text=query_texts, 
            images=query_image_inputs, 
            videos=query_video_inputs, 
            padding='longest', 
            return_tensors='pt'
        ).to(f"cuda:{self.device}")
        
        cache_position = torch.arange(0, len(query_texts))
        query_inputs = self.model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
        
        query_embeddings = self._get_embedding(output.hidden_states[-1], 1536)

        return query_embeddings, None

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 1, timeout_per_image: int = 60) -> np.ndarray:
        """
        Encode document images into single-vector embeddings.
        
        Note: DSE processes images one at a time due to chat template format.
        
        Args:
            images: List of PIL images
            batch_size: Not used (DSE is single-image)
            timeout_per_image: Skip images that take longer than this (seconds)
        """
        import signal
        import time
        
        all_embeddings = []
        skipped = 0
        
        # Get embedding dimension from a small test image
        test_img = Image.new('RGB', (28, 28))
        test_emb, _ = self._process_single_item(test_img)
        emb_dim = test_emb.shape[-1]
        
        for idx, img in enumerate(tqdm(images, desc="Encoding images (DSE)")):
            try:
                # Resize very large images to avoid memory issues
                img = img.convert("RGB")
                max_dim = 2048
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                start_time = time.time()
                embedding, _ = self._process_single_item(img)
                elapsed = time.time() - start_time
                
                if elapsed > timeout_per_image:
                    print(f"\n  Warning: Image {idx} took {elapsed:.1f}s", flush=True)
                
                all_embeddings.append(embedding.float().cpu().numpy())
                
            except Exception as e:
                print(f"\n  Error on image {idx}: {e}. Using zero embedding.", flush=True)
                all_embeddings.append(np.zeros((1, emb_dim), dtype=np.float32))
                skipped += 1
            
            torch.cuda.empty_cache()
        
        if skipped > 0:
            print(f"Skipped {skipped} images due to errors")
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 1) -> np.ndarray:
        """Encode text queries into single-vector embeddings.
        
        Note: Original HEAVEN does NOT normalize query embeddings at encoding time.
        Normalization happens at load time in utils.load_embedding().
        But since we use embeddings directly (not loading from file), we normalize here.
        """
        all_embeddings = []
        
        for query in tqdm(queries, desc="Encoding queries (DSE)"):
            embedding, _ = self._process_single_query(query)
            # _get_embedding already handles MRL slicing and normalization
            all_embeddings.append(embedding.float().cpu().numpy())
        
        return np.vstack(all_embeddings)


# =============================================================================
# ColQwen2.5 Encoder - Copied from HEAVEN/indexing/encode/colqwen25_encoder.py
# =============================================================================

class ColQwenEncoder(BaseEncoder):
    """
    ColQwen2.5 encoder implementation.
    
    Copied from: https://github.com/juyeonnn/HEAVEN/blob/main/indexing/encode/colqwen25_encoder.py
    """
    
    def __init__(self, device: str = "0", model_name: str = "vidore/colqwen2.5-v0.2"):
        self.model_name = model_name
        super().__init__(device)
    
    def _load_model(self, **kwargs):
        """Load ColQwen2.5 model and processor."""
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from transformers.utils.import_utils import is_flash_attn_2_available
        
        print(f"Loading ColQwen model: {self.model_name}", flush=True)
        
        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.device}",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        print(f"  Model loaded on cuda:{self.device}", flush=True)
        
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
        print("  Processor loaded!", flush=True)
    
    def _process_single_item(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single image and return embedding and input_ids."""
        # Process single image
        processed_image = self.processor.process_images([image]).to(self.model.device)
        
        with torch.no_grad():
            embedding = self.model(**processed_image)
        
        # Extract input_ids
        input_id = processed_image['input_ids'].cpu()
        
        return embedding.cpu(), input_id
    
    def _process_single_query(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single query and return embedding and input_ids."""
        # Process single query
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        
        with torch.no_grad():
            embedding = self.model(**batch_queries)
        
        # Extract input_ids
        input_id = batch_queries['input_ids'].cpu()
        
        return embedding.cpu(), input_id
    
    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 1) -> List[np.ndarray]:
        """
        Encode document images into multi-vector embeddings.
        
        Returns list of embeddings, one per image. Each embedding has shape (num_tokens, dim).
        
        Note: Original HEAVEN does NOT normalize at encoding time - normalization
        happens at load time in utils.load_embedding(). We normalize here since
        we use embeddings directly without saving/loading.
        """
        all_embeddings = []
        all_input_ids = []
        
        for img in tqdm(images, desc="Encoding images (ColQwen)"):
            img = img.convert("RGB")
            embedding, input_id = self._process_single_item(img)
            # Normalize - matching original HEAVEN's load_embedding behavior
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            all_embeddings.append(embedding.float().squeeze(0).numpy())
            all_input_ids.append(input_id)
            torch.cuda.empty_cache()
        
        return all_embeddings
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 1) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Encode text queries into multi-vector embeddings.
        
        Returns:
            Tuple of (embeddings_list, input_ids_list) where:
            - embeddings_list: list of embeddings, one per query, shape (num_tokens, dim)
            - input_ids_list: list of input_id tensors for token alignment
        
        Note: Original HEAVEN does NOT normalize at encoding time - normalization
        happens at load time in utils.load_embedding(). We normalize here since
        we use embeddings directly without saving/loading.
        """
        all_embeddings = []
        all_input_ids = []
        
        for query in tqdm(queries, desc="Encoding queries (ColQwen)"):
            embedding, input_id = self._process_single_query(query)
            # Normalize - matching original HEAVEN's load_embedding behavior
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            all_embeddings.append(embedding.float().squeeze(0).numpy())
            all_input_ids.append(input_id)
        
        return all_embeddings, all_input_ids


# =============================================================================
# SigLIP Encoder - Lightweight alternative (not in original HEAVEN)
# =============================================================================

class SigLIPEncoder(BaseEncoder):
    """
    SigLIP-based encoder as a lightweight alternative.
    
    Uses google/siglip-so400m-patch14-384 for fast visual encoding.
    Note: This is NOT in the original HEAVEN paper, provided as fallback.
    """
    
    def __init__(self, device: str = "0", model_name: str = "google/siglip-so400m-patch14-384"):
        self.model_name = model_name
        super().__init__(device)
    
    def _load_model(self, **kwargs):
        """Load SigLIP model and processor."""
        from transformers import AutoModel, AutoProcessor
        
        print(f"Loading SigLIP model: {self.model_name}", flush=True)
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to(f"cuda:{self.device}").eval()
        print("  Model loaded!", flush=True)
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("  Processor loaded!", flush=True)
    
    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode document images."""
        all_embeddings = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(images), batch_size),
                      total=num_batches,
                      desc="Encoding images (SigLIP)"):
            batch_images = images[i:i + batch_size]
            batch_images = [img.convert("RGB") for img in batch_images]
            
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(f"cuda:{self.device}")
            
            outputs = self.model.get_image_features(**inputs)
            embeddings = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            all_embeddings.append(embeddings.float().cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode text queries."""
        all_embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            inputs = self.processor(
                text=batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(f"cuda:{self.device}")
            
            outputs = self.model.get_text_features(**inputs)
            embeddings = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            all_embeddings.append(embeddings.float().cpu().numpy())
        
        return np.vstack(all_embeddings)


# =============================================================================
# Factory function and utilities
# =============================================================================

def get_encoder(encoder_type: str, device: str = "0") -> BaseEncoder:
    """Factory function to get encoder by type."""
    encoders = {
        "dse": DSEEncoder,
        "colqwen": ColQwenEncoder,
        "colqwen2.5": ColQwenEncoder,
        "colqwen25": ColQwenEncoder,
        "siglip": SigLIPEncoder,
    }
    
    if encoder_type.lower() not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(encoders.keys())}")
    
    return encoders[encoder_type.lower()](device=device)


def compute_maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """
    Compute MaxSim score between query and document multi-vector embeddings.
    
    Copied from HEAVEN scoring logic.
    
    Args:
        query_emb: Query embeddings of shape (num_query_tokens, dim)
        doc_emb: Document embeddings of shape (num_doc_tokens, dim)
    
    Returns:
        MaxSim score (higher is better)
    """
    # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
    similarities = np.dot(query_emb, doc_emb.T)
    
    # For each query token, take max over all doc tokens
    max_sims = similarities.max(axis=1)
    
    # Sum over all query tokens
    return float(max_sims.sum())
