"""
Attachment handler for multimodal AI support in Samosa GPT
Handles image and PDF attachments for use with Ollama vision models
"""
import os
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import io

# Import PDF processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not available. PDF text extraction disabled. Install with: pip install PyPDF2")

try:
    import pdf2image
    PDF_IMAGE_SUPPORT = True
except ImportError:
    PDF_IMAGE_SUPPORT = False
    logging.warning("pdf2image not available. PDF to image conversion disabled. Install with: pip install pdf2image")

logger = logging.getLogger(__name__)

class AttachmentHandler:
    """Handle image and PDF attachments for multimodal AI"""
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    SUPPORTED_PDF_FORMATS = {'.pdf'}
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB max file size
    MAX_IMAGE_DIMENSION = 2048  # Max width/height for images
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize attachment handler
        
        Args:
            cache_dir: Directory to cache processed attachments
        """
        self.cache_dir = cache_dir or Path("logs/attachments")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AttachmentHandler initialized with cache dir: {self.cache_dir}")
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate if file is supported and within size limits
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                size_mb = file_size / (1024 * 1024)
                return False, f"File too large: {size_mb:.1f}MB (max: {self.MAX_FILE_SIZE / (1024 * 1024)}MB)"
            
            # Check file extension
            ext = file_path.suffix.lower()
            if ext in self.SUPPORTED_IMAGE_FORMATS:
                return True, "Image file"
            elif ext in self.SUPPORTED_PDF_FORMATS:
                if not PDF_SUPPORT:
                    return False, "PDF support not available. Install PyPDF2: pip install PyPDF2"
                return True, "PDF file"
            else:
                supported = ', '.join(self.SUPPORTED_IMAGE_FORMATS | self.SUPPORTED_PDF_FORMATS)
                return False, f"Unsupported file format: {ext}. Supported: {supported}"
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False, f"Error validating file: {str(e)}"
    
    def process_image(self, image_path: Union[str, Path], resize: bool = True) -> Optional[Dict]:
        """Process an image file for multimodal AI
        
        Args:
            image_path: Path to the image file
            resize: Whether to resize large images
            
        Returns:
            Dictionary with image data and metadata, or None on error
        """
        try:
            image_path = Path(image_path)
            
            # Validate file
            is_valid, message = self.validate_file(image_path)
            if not is_valid:
                logger.error(f"Invalid image file: {message}")
                return None
            
            # Open and process image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                original_size = img.size
                
                # Resize if needed
                if resize and (img.width > self.MAX_IMAGE_DIMENSION or img.height > self.MAX_IMAGE_DIMENSION):
                    # Calculate new size maintaining aspect ratio
                    ratio = min(self.MAX_IMAGE_DIMENSION / img.width, self.MAX_IMAGE_DIMENSION / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {original_size} to {new_size}")
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return {
                    'type': 'image',
                    'format': 'png',
                    'data': img_base64,
                    'size': img.size,
                    'original_size': original_size,
                    'file_name': image_path.name,
                    'file_path': str(image_path)
                }
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def process_pdf(self, pdf_path: Union[str, Path], extract_text: bool = True, 
                    extract_images: bool = True, max_pages: int = 10) -> Optional[Dict]:
        """Process a PDF file for multimodal AI
        
        Args:
            pdf_path: Path to the PDF file
            extract_text: Whether to extract text content
            extract_images: Whether to extract images from PDF
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary with PDF data and metadata, or None on error
        """
        try:
            pdf_path = Path(pdf_path)
            
            # Validate file
            is_valid, message = self.validate_file(pdf_path)
            if not is_valid:
                logger.error(f"Invalid PDF file: {message}")
                return None
            
            result = {
                'type': 'pdf',
                'file_name': pdf_path.name,
                'file_path': str(pdf_path),
                'pages': [],
                'text_content': '',
                'images': []
            }
            
            # Extract text using PyPDF2
            if extract_text and PDF_SUPPORT:
                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        num_pages = min(len(pdf_reader.pages), max_pages)
                        result['total_pages'] = len(pdf_reader.pages)
                        result['processed_pages'] = num_pages
                        
                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            result['pages'].append({
                                'page_num': page_num + 1,
                                'text': text
                            })
                            result['text_content'] += f"\n--- Page {page_num + 1} ---\n{text}\n"
                        
                        logger.info(f"Extracted text from {num_pages} pages of {pdf_path.name}")
                except Exception as e:
                    logger.error(f"Error extracting text from PDF: {e}")
            
            # Convert PDF pages to images
            if extract_images and PDF_IMAGE_SUPPORT:
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=min(max_pages, 5))
                    
                    for i, img in enumerate(images):
                        # Resize if needed
                        if img.width > self.MAX_IMAGE_DIMENSION or img.height > self.MAX_IMAGE_DIMENSION:
                            ratio = min(self.MAX_IMAGE_DIMENSION / img.width, self.MAX_IMAGE_DIMENSION / img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        result['images'].append({
                            'page_num': i + 1,
                            'data': img_base64,
                            'size': img.size
                        })
                    
                    logger.info(f"Converted {len(images)} pages to images from {pdf_path.name}")
                except Exception as e:
                    logger.error(f"Error converting PDF to images: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def process_attachment(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """Process any supported attachment file
        
        Args:
            file_path: Path to the attachment file
            
        Returns:
            Dictionary with attachment data and metadata, or None on error
        """
        try:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()
            
            if ext in self.SUPPORTED_IMAGE_FORMATS:
                return self.process_image(file_path)
            elif ext in self.SUPPORTED_PDF_FORMATS:
                return self.process_pdf(file_path)
            else:
                logger.error(f"Unsupported file type: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing attachment {file_path}: {e}")
            return None
    
    def get_multimodal_models(self) -> List[str]:
        """Get list of Ollama models that support vision/multimodal input
        
        Returns:
            List of multimodal model names
        """
        # Common Ollama vision models
        return [
            'llava',
            'llava:13b',
            'llava:34b',
            'llava-llama3',
            'llava-phi3',
            'bakllava',
            'moondream',
            'cogvlm',
            'cogagent'
        ]
    
    def format_for_ollama(self, prompt: str, attachments: List[Dict]) -> Dict:
        """Format prompt and attachments for Ollama API
        
        Args:
            prompt: Text prompt
            attachments: List of processed attachment dictionaries
            
        Returns:
            Dictionary formatted for Ollama chat API
        """
        try:
            # Build messages with images
            images = []
            context_parts = [prompt]
            
            for attachment in attachments:
                if attachment['type'] == 'image':
                    # Add image to images list for Ollama
                    images.append(attachment['data'])
                    context_parts.append(f"[Image attached: {attachment['file_name']}]")
                    
                elif attachment['type'] == 'pdf':
                    # Add PDF text content to prompt context
                    if attachment.get('text_content'):
                        context_parts.append(f"\n[PDF Document: {attachment['file_name']}]")
                        context_parts.append(attachment['text_content'])
                    
                    # Add PDF images if available
                    if attachment.get('images'):
                        for img in attachment['images'][:3]:  # Limit to first 3 pages
                            images.append(img['data'])
                            context_parts.append(f"[PDF Page {img['page_num']} as image]")
            
            # Combine context
            full_prompt = "\n\n".join(context_parts)
            
            result = {
                'prompt': full_prompt,
                'has_images': len(images) > 0
            }
            
            if images:
                result['images'] = images
            
            return result
            
        except Exception as e:
            logger.error(f"Error formatting for Ollama: {e}")
            return {'prompt': prompt, 'has_images': False}
    
    def clear_cache(self) -> bool:
        """Clear the attachment cache directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Attachment cache cleared")
                return True
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Create global instance
attachment_handler = AttachmentHandler()
