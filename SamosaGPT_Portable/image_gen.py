import streamlit as st
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
import io
import zipfile
import time
from PIL import Image
import numpy as np

# Import custom component
from streamlit_markdown_select import markdown_select, create_option

# Import hardware optimizer
try:
    from hardware_optimizer import get_image_settings, get_hardware_info, get_hardware_optimizer
    HARDWARE_OPTIMIZER_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZER_AVAILABLE = False

# Import user preferences
try:
    from user_preferences import get_preferences_manager
    _preferences_manager = get_preferences_manager()
    PREFERENCES_AVAILABLE = True
except ImportError:
    PREFERENCES_AVAILABLE = False
    _preferences_manager = None

# Import image model manager
try:
    from image_model_manager import image_model_manager
    IMAGE_MODEL_MANAGER_AVAILABLE = True
except ImportError:
    IMAGE_MODEL_MANAGER_AVAILABLE = False
    image_model_manager = None

# Available models - now managed by ImageModelManager
AVAILABLE_MODELS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
    "Stable Diffusion XL Base": "stabilityai/stable-diffusion-xl-base-1.0",
    "Dreamlike Photoreal 2.0": "dreamlike-art/dreamlike-photoreal-2.0",
    "Realistic Vision v6.0": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "Anything v5": "stablediffusionapi/anything-v5",
    "OpenJourney": "prompthero/openjourney",
    "Deliberate v2": "XpucT/Deliberate"
}

# Load the Stable Diffusion model
@st.cache_resource
def load_model(model_id):
    """Load and cache the diffusion model"""
    try:
        st.info(f"Loading model: {model_id}")
        
        # Try to load as StableDiffusionPipeline first
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
        except Exception:
            # Fallback to generic DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
        
        # Optimize memory usage
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_sequential_cpu_offload'):
            if torch.cuda.is_available():
                pipeline.enable_sequential_cpu_offload()
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            if torch.cuda.is_available():
                pipeline.enable_model_cpu_offload()
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(pipeline, 'enable_sequential_cpu_offload'):
            pipeline = pipeline.to(device)
        
        st.success(f"✅ Model loaded on {device.upper()}")
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        st.error("This might be due to:")
        st.error("- Network connectivity issues")
        st.error("- Invalid model ID")
        st.error("- Insufficient memory")
        st.error("- Missing dependencies")
        st.error("- Model not compatible with current diffusers version")
        return None

def generate_single_image(pipeline, prompt, params, seed_offset=0):
    """Generate a single image with given parameters"""
    try:
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "num_inference_steps": params["steps"],
            "guidance_scale": params["guidance"],
            "width": params["width"],
            "height": params["height"],
        }
        
        # Handle seed
        if params["seed"] is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(params["seed"] + seed_offset)
            generation_kwargs["generator"] = generator
        
        # Generate image
        with torch.no_grad():
            result = pipeline(**generation_kwargs)
            
        # Extract image
        if hasattr(result, 'images'):
            image = result.images[0]
        else:
            image = result[0]
            
        return image, None
        
    except Exception as e:
        return None, str(e)

def create_image_grid(images, max_cols=3):
    """Create a grid layout for displaying multiple images"""
    num_images = len(images)
    
    if num_images == 1:
        st.image(images[0], caption="Generated Image", use_container_width=True)
    elif num_images == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(images[0], caption="Image 1", use_container_width=True)
        with col2:
            st.image(images[1], caption="Image 2", use_container_width=True)
    else:
        # For 3+ images, use dynamic grid
        cols_per_row = min(max_cols, num_images)
        rows = (num_images + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < num_images:
                    with cols[col_idx]:
                        st.image(images[img_idx], caption=f"Image {img_idx + 1}", use_container_width=True)

def create_download_section(images, model_name):
    """Create download buttons for images"""
    st.subheader("📥 Download Options")
    
    # Individual downloads
    if len(images) <= 4:
        cols = st.columns(len(images))
        for i, (image, col) in enumerate(zip(images, cols)):
            with col:
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"Download Image {i+1}",
                    data=img_buffer.getvalue(),
                    file_name=f"generated_image_{i+1}_{model_name.replace(' ', '_').replace(':', '_')}.png",
                    mime="image/png",
                    key=f"download_single_{i}"
                )
    else:
        # For many images, create a simple list
        for i, image in enumerate(images):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label=f"📄 Download Image {i+1}",
                data=img_buffer.getvalue(),
                file_name=f"generated_image_{i+1}_{model_name.replace(' ', '_').replace(':', '_')}.png",
                mime="image/png",
                key=f"download_single_{i}"
            )
    
    # ZIP download for multiple images
    if len(images) > 1:
        st.markdown("---")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, image in enumerate(images):
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zip_file.writestr(
                    f"generated_image_{i+1}_{model_name.replace(' ', '_').replace(':', '_')}.png",
                    img_buffer.getvalue()
                )
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download All Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"generated_images_{model_name.replace(' ', '_').replace(':', '_')}.zip",
            mime="application/zip",
            key="download_zip"
        )

def main():
    """Main image generation interface"""
    st.title("🎨 AI Image Generator")
    st.markdown("Generate stunning images using state-of-the-art AI models")
    
    # Get hardware-optimized settings
    if HARDWARE_OPTIMIZER_AVAILABLE:
        image_settings = get_image_settings()
        hardware_info = get_hardware_info()
        optimizer = get_hardware_optimizer()
        
        # Show hardware performance indicator
        perf_rec = optimizer.get_performance_recommendation()
        st.info(f"🖥️ Hardware Status: {perf_rec}")
        
        # Use optimized defaults
        default_num_images = image_settings.get('default_num_images', 2)
        max_images = image_settings.get('max_images', 10)
        default_resolution = image_settings.get('recommended_resolution', [512, 512])
        max_resolution = image_settings.get('max_resolution', [768, 768])
        default_steps = image_settings.get('inference_steps', 20)
        default_guidance = image_settings.get('guidance_scale', 7.5)
        enable_attention_slicing = image_settings.get('enable_attention_slicing', True)
        enable_cpu_offload = image_settings.get('enable_cpu_offload', False)
        recommended_batch_size = image_settings.get('batch_size', 1)
    else:
        # Fallback defaults
        default_num_images = 2
        max_images = 10
        default_resolution = [512, 512]
        max_resolution = [768, 768]
        default_steps = 20
        default_guidance = 7.5
        enable_attention_slicing = True
        enable_cpu_offload = False
        recommended_batch_size = 1

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Available Models:**
        - **Stable Diffusion v1.5**: General purpose, reliable
        - **Stable Diffusion v2.1**: Improved quality and coherence
        - **Stable Diffusion XL**: High resolution, detailed
        - **Dreamlike Photoreal**: Photorealistic images
        - **Realistic Vision**: Ultra-realistic outputs
        - **Anything v5**: Anime and artistic styles
        - **OpenJourney**: Midjourney-style artistic
        - **Deliberate**: Balanced artistic style
        """)
        
        st.header("💡 Pro Tips")
        st.markdown("""
        - **Be descriptive**: "A red sports car on a mountain road at sunset"
        - **Add style**: "photorealistic", "oil painting", "digital art"
        - **Quality terms**: "highly detailed", "8k", "masterpiece"
        - **Lighting**: "soft lighting", "dramatic shadows"
        - **Generate multiple**: Try 2-4 images for variety
        - **Use seeds**: For reproducible results
        """)
        
        # Hardware-optimized system info
        st.header("🖥️ System Status")
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.write(f"**Device:** {device}")
        
        if HARDWARE_OPTIMIZER_AVAILABLE and hardware_info.get('gpu', {}).get('cuda_available'):
            gpu_info = hardware_info['gpu']
            vram_gb = gpu_info.get('total_vram_gb', 0)
            
            st.write(f"**GPU:** {gpu_info.get('gpus', [{}])[0].get('name', 'Unknown')}")
            st.write(f"**VRAM:** {vram_gb:.1f} GB")
            
            # Hardware-based recommendations
            if vram_gb >= 16:
                st.success("🚀 Excellent GPU - High resolution, multiple images")
                st.info(f"💡 Recommended: {max_images} images at {max_resolution[0]}x{max_resolution[1]}")
            elif vram_gb >= 12:
                st.success("✅ Great GPU - Good resolution, multiple images")
                st.info(f"💡 Recommended: {default_num_images} images at {default_resolution[0]}x{default_resolution[1]}")
            elif vram_gb >= 8:
                st.info("👍 Good GPU - Standard resolution")
                st.info(f"💡 Recommended: {default_num_images} images at {default_resolution[0]}x{default_resolution[1]}")
            elif vram_gb >= 4:
                st.warning("⚠️ Limited VRAM - Use lower resolution")
                st.info(f"💡 Recommended: 1-2 images at 512x512")
            else:
                st.error("❌ Very limited VRAM - Consider CPU mode")
        elif torch.cuda.is_available():
            # Fallback for when hardware optimizer is not available
            gpu_name = torch.cuda.get_device_name()
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
            st.write(f"**GPU:** {gpu_name}")
            st.write(f"**VRAM:** {vram} GB")
            
            if vram >= 8:
                st.success("✅ High-end GPU detected")
            elif vram >= 6:
                st.info("ℹ️ Good GPU for most models")
            else:
                st.warning("⚠️ Limited VRAM - use smaller resolutions")
        else:
            st.warning("⚠️ No GPU detected - generation will be slow")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection with user preferences using custom component
        image_prefs = {}
        if _preferences_manager:
            image_prefs = _preferences_manager.get_category_preferences('image_generation')
        
        # Get available models from manager if available
        if IMAGE_MODEL_MANAGER_AVAILABLE:
            available_models = image_model_manager.get_available_models()
            installed_models = image_model_manager.get_installed_models()
        else:
            available_models = AVAILABLE_MODELS
            installed_models = list(AVAILABLE_MODELS.keys())  # Assume all are installed
        
        # Load preferred model
        preferred_model = image_prefs.get('preferred_model') or image_prefs.get('last_used_model')
        
        # Handle cases where preferred_model might be a dict (backward compatibility)
        if isinstance(preferred_model, dict):
            preferred_model = preferred_model.get('value')
        
        # Set default model (first one in the list)
        default_model = "Stable Diffusion v1.5"
        
        # Create options for the custom component
        model_options = []
        for model_name in available_models.keys():
            is_installed = model_name in installed_models if IMAGE_MODEL_MANAGER_AVAILABLE else True
            is_default = model_name == default_model
            
            # Create buttons based on installation status
            buttons = []
            if is_installed:
                buttons.append({"text": "🗑️ Delete", "variant": "danger", "action": "delete"})
            else:
                buttons.append({"text": "📥 Install", "variant": "primary", "action": "install"})
            
            if is_default:
                # Add green indicator for default model
                option = create_option(
                    value=model_name,
                    label=f"🤖 {model_name}" + (" ✅" if is_installed else " 📥"),
                    code="default" if is_installed else "not installed",
                    code_color="#00cc88" if is_installed else "#FF9800",
                    buttons=buttons
                )
            else:
                option = create_option(
                    value=model_name,
                    label=f"🤖 {model_name}" + (" ✅" if is_installed else " 📥"),
                    code=f"Model: {model_name}" + (" (Installed)" if is_installed else " (Not Installed)"),
                    language="text",
                    code_color="#00cc88" if is_installed else "#FF9800",
                    buttons=buttons
                )
            model_options.append(option)
        
        # Add option to add custom model
        custom_option = create_option(
            value="__add_custom__",
            label="➕ Add Custom Model",
            code="Add model from Hugging Face",
            language="info",
            code_color="#2196F3",
            buttons=[
                {"text": "🔍 Browse", "variant": "primary", "action": "browse"},
                {"text": "➕ Add", "variant": "secondary", "action": "add"}
            ]
        )
        model_options.append(custom_option)
        
        # Use preferred model or default to first model
        default_value = preferred_model if preferred_model and preferred_model in available_models.keys() else default_model
        
        st.markdown("**🤖 Choose AI Model:**")
        model_selection = markdown_select(
            options=model_options,
            default_value=default_value,
            key="image_model_selection"
        )
        
        # Handle button clicks and model selection
        selected_model_name = None
        
        # Check if we should show custom model interface
        show_custom = False
        if st.session_state.get('show_custom_model_interface'):
            show_custom = True
            # Clear the flag after using it
            st.session_state['show_custom_model_interface'] = False
        
        # Extract the actual selection value from the component
        actual_selection = None
        if model_selection:
            if isinstance(model_selection, dict):
                # Handle dictionary response from component
                if model_selection.get('type') == 'selection':
                    actual_selection = model_selection.get('value')
                elif model_selection.get('type') == 'button_click':
                    # This is a button click, not a selection
                    actual_selection = None
                else:
                    actual_selection = model_selection.get('value', model_selection)
            else:
                actual_selection = model_selection
        
        # Check if add custom model is selected
        if actual_selection == "__add_custom__":
            show_custom = True
        
        if show_custom:
            # Show custom model interface
            st.subheader("➕ Add Custom Model")
            
            col1_custom, col2_custom = st.columns([3, 1])
            with col1_custom:
                custom_model_name = st.text_input(
                    "Model Name",
                    placeholder="e.g., My Custom Model",
                    help="Enter a friendly name for the model"
                )
                custom_model_id = st.text_input(
                    "Hugging Face Model ID",
                    placeholder="e.g., username/model-name",
                    help="Enter the Hugging Face model ID (e.g., 'runwayml/stable-diffusion-v1-5')"
                )
            with col2_custom:
                if st.button("➕ Add Model", type="primary"):
                    if custom_model_name and custom_model_id:
                        if IMAGE_MODEL_MANAGER_AVAILABLE:
                            with st.spinner(f"Adding {custom_model_name}..."):
                                success = image_model_manager.add_custom_model(custom_model_name, custom_model_id)
                                
                                # Clear custom interface flag
                                if 'show_custom_model_interface' in st.session_state:
                                    del st.session_state['show_custom_model_interface']
                                
                                if success:
                                    st.success(f"✅ {custom_model_name} added successfully!")
                                    # Update AVAILABLE_MODELS for immediate use
                                    AVAILABLE_MODELS[custom_model_name] = custom_model_id
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to add {custom_model_name}")
                        else:
                            st.error("Model manager not available")
                    else:
                        st.error("Please enter both model name and ID")
            
            # Show popular models as suggestions
            st.subheader("💡 Popular Models")
            popular_models = [
                ("Stable Diffusion v2.1", "stabilityai/stable-diffusion-2-1"),
                ("Stable Diffusion XL", "stabilityai/stable-diffusion-xl-base-1.0"),
                ("Dreamlike Photoreal", "dreamlike-art/dreamlike-photoreal-2.0"),
                ("Realistic Vision", "SG161222/Realistic_Vision_V6.0_B1_noVAE"),
                ("Anything v5", "stablediffusionapi/anything-v5"),
                ("OpenJourney", "prompthero/openjourney")
            ]
            
            cols = st.columns(3)
            for i, (model_name, model_id) in enumerate(popular_models):
                with cols[i % 3]:
                    if model_name not in available_models:
                        if st.button(f"➕ {model_name}", key=f"add_{model_name}"):
                            if IMAGE_MODEL_MANAGER_AVAILABLE:
                                with st.spinner(f"Adding {model_name}..."):
                                    success = image_model_manager.add_custom_model(model_name, model_id)
                                    
                                    # Clear custom interface flag
                                    if 'show_custom_model_interface' in st.session_state:
                                        del st.session_state['show_custom_model_interface']
                                    
                                    if success:
                                        st.success(f"✅ {model_name} added successfully!")
                                        # Update AVAILABLE_MODELS for immediate use
                                        AVAILABLE_MODELS[model_name] = model_id
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to add {model_name}")
            
            # Set model to first available for now
            selected_model_name = list(available_models.keys())[0] if available_models else default_model
        elif actual_selection:
            selected_model_name = actual_selection
        else:
            # If no model selected, use first available or default
            selected_model_name = list(available_models.keys())[0] if available_models else default_model
        
        # Handle button clicks from component
        component_value = st.session_state.get('image_model_selection')
        if component_value and isinstance(component_value, dict) and component_value.get('type') == 'button_click':
            button_data = component_value.get('button_data')
            if button_data:
                action = button_data.get('action')
                option_value = button_data.get('optionValue')
                
                if action == "delete" and option_value and IMAGE_MODEL_MANAGER_AVAILABLE:
                    # Use a separate session state key for delete confirmation
                    delete_key = f"confirm_delete_image_{option_value}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        st.warning(f"Are you sure you want to delete model '{option_value}'?")
                        # Show model size if available
                        model_size = image_model_manager.get_model_size(option_value)
                        if model_size:
                            st.info(f"Model size: {model_size}")
                        
                        col1_del, col2_del = st.columns([1, 1])
                        with col1_del:
                            if st.button("🗑️ Yes, Delete", type="primary", key=f"btn_confirm_img_{option_value}"):
                                st.session_state[delete_key] = True
                                st.rerun()
                        with col2_del:
                            if st.button("❌ Cancel", key=f"btn_cancel_img_{option_value}"):
                                # Reset any confirmation states
                                for key in list(st.session_state.keys()):
                                    if key.startswith('confirm_delete_image_'):
                                        del st.session_state[key]
                                st.rerun()
                    else:
                        # Perform the deletion
                        with st.spinner(f"Deleting {option_value}..."):
                            success = image_model_manager.delete_model(option_value)
                            
                            # Clear the confirmation state first
                            if delete_key in st.session_state:
                                del st.session_state[delete_key]
                            
                            # Clear any other model-related session state
                            keys_to_remove = [key for key in st.session_state.keys() if key.startswith('image_model_selection')]
                            for key in keys_to_remove:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            # Clear pipeline cache if current model was deleted
                            if option_value == selected_model_name:
                                if 'current_model_id' in st.session_state:
                                    del st.session_state['current_model_id']
                                if 'pipeline' in st.session_state:
                                    del st.session_state['pipeline']
                            
                            if success:
                                st.success(f"✅ {option_value} deleted successfully!")
                                # Force rerun to refresh the model list
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete {option_value}")
                
                elif action == "install" and option_value and IMAGE_MODEL_MANAGER_AVAILABLE:
                    with st.spinner(f"Installing {option_value}..."):
                        success = image_model_manager.install_model(option_value)
                        
                        # Clear any model-related session state
                        keys_to_remove = [key for key in st.session_state.keys() if key.startswith('image_model_selection')]
                        for key in keys_to_remove:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        if success:
                            st.success(f"✅ {option_value} installed successfully!")
                            # Force rerun to refresh the model list
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to install {option_value}")
                
                elif action == "browse":
                    # Set flag to show custom model interface
                    st.session_state['show_custom_model_interface'] = True
                    st.rerun()
                
                elif action == "add":
                    # Set flag to show custom model interface
                    st.session_state['show_custom_model_interface'] = True
                    st.rerun()
        
        # Fallback to first model if nothing selected
        if not selected_model_name and available_models:
            selected_model_name = list(available_models.keys())[0]
        
        # Handle different return types from markdown_select
        if selected_model_name is None:
            selected_model_name = default_value
        elif isinstance(selected_model_name, dict):
            # Extract value from dictionary response (component returns {type: 'selection', value: 'actual_value'})
            selected_model_name = selected_model_name.get('value', default_value)
        
        # Ensure we have a valid model name
        if not selected_model_name or selected_model_name not in available_models:
            selected_model_name = default_value
        
        # Save model preference when changed
        current_last_used = image_prefs.get('last_used_model')
        if isinstance(current_last_used, dict):
            current_last_used = current_last_used.get('value')
        
        if _preferences_manager and selected_model_name != current_last_used:
            _preferences_manager.set_preference('image_generation', 'last_used_model', selected_model_name)
            _preferences_manager.save_preferences()
        
        if "prompt" not in st.session_state:
            st.session_state["prompt"] = ""

        # Prompt input
        prompt = st.text_area(
            "✍️ Describe your image:",
            value = st.session_state["prompt"],
            placeholder="A beautiful landscape with mountains and a lake at golden hour...",
            height=100,
            help="Be descriptive! The more details you provide, the better your image will be."
        )
        
        # Quick style buttons
        st.markdown("**🎨 Quick Styles:**")
        style_col1, style_col2, style_col3, style_col4 = st.columns(4)
        with style_col1:
            if st.button("📸 Photorealistic"):
                if prompt != "":
                    st.session_state["prompt"] = f"{prompt}, photorealistic, highly detailed, 8k resolution"
                    st.rerun()
        with style_col2:
            if st.button("🎭 Artistic"):
                if prompt != "":
                    st.session_state["prompt"] = f"{prompt}, digital art, artstation, concept art"
                    st.rerun()
        with style_col3:
            if st.button("🌸 Anime"):
                if prompt != "":
                    st.session_state["prompt"] = f"{prompt}, anime style, studio ghibli, detailed"
                    st.rerun()
        with style_col4:
            if st.button("🖼️ Oil Painting"):
                if prompt != "":
                    st.session_state["prompt"] = f"{prompt}, oil painting, classical art style"
                    st.rerun()
    
    with col2:
        st.markdown("### ⚙️ Generation Settings")
        # Number of images with preferences
        preferred_num_images = image_prefs.get('preferred_num_images', default_num_images)
        num_images = st.number_input(
            "🔢 Number of Images", 
            min_value=1, max_value=max_images, value=preferred_num_images,
            help=f"Generate multiple images for variety (optimized default: {default_num_images})"
        )
        
        # Save num_images preference when changed
        if _preferences_manager and num_images != image_prefs.get('preferred_num_images'):
            _preferences_manager.set_preference('image_generation', 'preferred_num_images', num_images)
            _preferences_manager.save_preferences()
        
        # Image dimensions with preferences
        resolution_options = []
        if max_resolution[0] >= 1024:
            resolution_options.append("1024x1024 (High Quality)")
        if max_resolution[0] >= 768:
            resolution_options.append("768x768 (Balanced)")
        resolution_options.append("512x512 (Fast)")
        
        # Load preferred resolution
        preferred_resolution = image_prefs.get('preferred_resolution', f"{default_resolution[0]}x{default_resolution[1]}")
        
        # Handle cases where preferred_resolution might be a list (backward compatibility)
        if isinstance(preferred_resolution, list) and len(preferred_resolution) >= 2:
            preferred_resolution = f"{preferred_resolution[0]}x{preferred_resolution[1]}"
        elif not isinstance(preferred_resolution, str):
            preferred_resolution = f"{default_resolution[0]}x{default_resolution[1]}"
        default_idx = 0
        
        # Map preferred resolution to index
        resolution_map = {
            "1024x1024": "1024x1024 (High Quality)",
            "768x768": "768x768 (Balanced)", 
            "512x512": "512x512 (Fast)"
        }
        
        preferred_res_text = resolution_map.get(preferred_resolution, "512x512 (Fast)")
        if preferred_res_text in resolution_options:
            default_idx = resolution_options.index(preferred_res_text)
        
        resolution = st.selectbox(
            "📐 Resolution",
            options=resolution_options,
            index=default_idx,
            help=f"Optimized for your hardware: {default_resolution[0]}x{default_resolution[1]} recommended"
        )
        
        # Save resolution preference when changed
        resolution_key = resolution.split(" (")[0]  # Extract "512x512" from "512x512 (Fast)"
        if _preferences_manager and resolution_key != image_prefs.get('preferred_resolution'):
            _preferences_manager.set_preference('image_generation', 'preferred_resolution', resolution_key)
            _preferences_manager.save_preferences()
        
        width, height = {
            "512x512 (Fast)": (512, 512),
            "768x768 (Balanced)": (768, 768),
            "1024x1024 (High Quality)": (1024, 1024)
        }[resolution]

    # Advanced options
    with st.expander("🔧 Advanced Settings (Hardware Optimized)"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            preferred_steps = image_prefs.get('preferred_steps', default_steps)
            preferred_guidance = image_prefs.get('preferred_guidance', default_guidance)
            
            steps = st.slider(
                "Inference Steps", 
                min_value=10, max_value=100, value=preferred_steps,
                help=f"Optimized default: {default_steps} steps for your hardware"
            )
            
            # Save steps preference when changed
            if _preferences_manager and steps != image_prefs.get('preferred_steps'):
                _preferences_manager.set_preference('image_generation', 'preferred_steps', steps)
                _preferences_manager.save_preferences()
            
            guidance = st.slider(
                "Guidance Scale", 
                min_value=1.0, max_value=20.0, value=preferred_guidance, step=0.5,
                help="Higher values = follows prompt more closely"
            )
            
            # Save guidance preference when changed
            if _preferences_manager and guidance != image_prefs.get('preferred_guidance'):
                _preferences_manager.set_preference('image_generation', 'preferred_guidance', guidance)
                _preferences_manager.save_preferences()
            
        with col_adv2:
            use_seed = st.checkbox("🎲 Use Fixed Seed", help="For reproducible results")
            seed = None
            if use_seed:
                seed = st.number_input(
                    "Seed Value", 
                    min_value=0, max_value=2147483647, value=42,
                    help="Same seed + prompt = same image"
                )
            
            # Show model management options
            if IMAGE_MODEL_MANAGER_AVAILABLE:
                st.markdown("**📊 Model Info:**")
                if image_model_manager.is_model_installed(selected_model_name):
                    model_size = image_model_manager.get_model_size(selected_model_name)
                    st.info(f"📁 Size: {model_size or 'Unknown'}")
                    
                    # Show installed models count
                    installed_count = len(image_model_manager.get_installed_models())
                    total_count = len(image_model_manager.get_available_models())
                    st.info(f"📚 Installed: {installed_count}/{total_count} models")
                else:
                    st.warning("⚠️ Model not installed locally")
            
            # Custom model option (legacy support)
            use_custom = st.checkbox("🔧 Custom Model", help="Use any Hugging Face model")
            if use_custom:
                custom_model = st.text_input(
                    "Model ID:", 
                    placeholder="e.g., runwayml/stable-diffusion-v1-5"
                )
                if custom_model:
                    selected_model_name = f"Custom: {custom_model}"
                    available_models[selected_model_name] = custom_model
                    # Update the selected model ID
                    selected_model_id = custom_model

    # Load model
    # Additional safety check to ensure we have a valid model
    if not selected_model_name or selected_model_name not in available_models:
        selected_model_name = "Stable Diffusion v1.5"  # Default fallback
        
    selected_model_id = available_models[selected_model_name]
    
    # Check if model is installed (if manager is available)
    if IMAGE_MODEL_MANAGER_AVAILABLE:
        if not image_model_manager.is_model_installed(selected_model_name):
            st.error(f"❌ Model '{selected_model_name}' is not installed. Please install it first.")
            st.info("Click the 📥 Install button next to the model to download it.")
            st.stop()
    
    # Model loading logic
    if 'current_model_id' not in st.session_state:
        st.session_state.current_model_id = None
        st.session_state.pipeline = None
        
    if st.session_state.current_model_id != selected_model_id:
        with st.spinner(f"🔄 Loading {selected_model_name}..."):
            st.session_state.pipeline = load_model(selected_model_id)
            st.session_state.current_model_id = selected_model_id if st.session_state.pipeline else None

    pipeline = st.session_state.pipeline

    if pipeline is None:
        st.error("❌ Failed to load the selected model. Please try a different model.")
        st.stop()

    st.success(f"✅ {selected_model_name} is ready!")

    # Generation button
    if st.button("🚀 Generate Images", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate images.")
            return
            
        # Prepare parameters
        params = {
            "steps": steps,
            "guidance": guidance,
            "width": width,
            "height": height,
            "seed": seed
        }
        
        # Generate images
        with st.spinner(f"🎨 Generating {num_images} image{'s' if num_images > 1 else ''}..."):
            images = []
            errors = []
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                for i in range(num_images):
                    status_text.text(f"Generating image {i+1} of {num_images}...")
                    progress_bar.progress(i / num_images)
                    
                    # Generate single image
                    image, error = generate_single_image(pipeline, prompt, params, seed_offset=i)
                    
                    if image:
                        images.append(image)
                    else:
                        errors.append(f"Image {i+1}: {error}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / num_images)
                
                status_text.text("Generation complete!")
                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                status_text.empty()
        
        # Display results
        if images:
            st.success(f"✅ Successfully generated {len(images)} image{'s' if len(images) > 1 else ''}!")
            
            # Show images
            st.markdown("### 🖼️ Generated Images")
            create_image_grid(images)
            
            # Download section
            create_download_section(images, selected_model_name)
            
            # Generation info
            with st.expander("📊 Generation Details"):
                generation_info = {
                    "model": selected_model_name,
                    "model_id": selected_model_id,
                    "prompt": prompt,
                    "images_generated": len(images),
                    "resolution": f"{width}x{height}",
                    "steps": steps,
                    "guidance_scale": guidance,
                    "seed": seed if seed is not None else "Random",
                    "errors": len(errors)
                }
                
                # Add model management info if available
                if IMAGE_MODEL_MANAGER_AVAILABLE:
                    generation_info["model_installed"] = image_model_manager.is_model_installed(selected_model_name)
                    model_size = image_model_manager.get_model_size(selected_model_name)
                    if model_size:
                        generation_info["model_size"] = model_size
                
                st.json(generation_info)
        
        # Show errors if any
        if errors:
            st.error("⚠️ Some images failed to generate:")
            for error in errors:
                st.error(error)
        
        if not images:
            st.error("❌ Failed to generate any images. Please try again with different settings.")


# Run standalone if executed directly
if __name__ == "__main__":
    main()