# Color Scheme Changes for Aurora Streamlit Website

## New Color Scheme Applied

The following color scheme has been applied to the Aurora Streamlit website:

```json
{
  "colors": {
    "primary": "#6B46C1",      // Purple - Main brand color
    "secondary": "#3B82F6",    // Blue - Secondary interactions
    "accent1": "#34D399",      // Green - Success/positive actions
    "accent2": "#EC4899",      // Pink - Warnings/delete actions
    "background": "#0F172A"    // Dark slate - Main background
  }
}
```

## Files Modified

### 1. `.streamlit/config.toml` (Created)
- Set Streamlit theme colors to match the new scheme
- Primary color: #6B46C1 (Purple)
- Background color: #0F172A (Dark slate)
- Secondary background: #1E293B (Lighter slate)
- Text color: #F1F5F9 (Light gray)

### 2. `streamlit_navbar/frontend/src/index.css`
- Updated CSS variables for dark mode navbar
  - Background: #0F172A
  - Text: #F1F5F9
  - Border: rgba(59, 130, 246, 0.2) (Blue tint)
  - Hover: rgba(107, 70, 193, 0.2) (Purple tint)
  - Active: #6B46C1 (Purple)
- Added new CSS variables for color scheme
  - `--primary-color: #6B46C1`
  - `--secondary-color: #3B82F6`
  - `--accent1-color: #34D399`
  - `--accent2-color: #EC4899`
  - `--background-color: #0F172A`

### 3. `streamlit_navbar/__init__.py`
- Updated default navbar style
  - Background: #0F172A
  - Text color: #F1F5F9
  - Border: Blue-tinted with purple shadow
  - Box shadow: Purple-tinted

### 4. `streamlit_markdown_select/frontend/src/index.css`
- Updated primary color references to #6B46C1 (Purple)
- Updated dark mode colors:
  - Background: #0F172A
  - Border: #3B82F6 (Blue)
  - Text: #F1F5F9
  - Secondary background: #1E293B
- Updated button colors:
  - Install button: #34D399 (Green - accent1)
  - Delete button: #EC4899 (Pink - accent2)
  - Browse button: #3B82F6 (Blue - secondary)

## Color Usage Guidelines

- **Primary (#6B46C1 - Purple)**: Main brand color, active states, primary CTAs
- **Secondary (#3B82F6 - Blue)**: Secondary interactions, borders, links
- **Accent1 (#34D399 - Green)**: Success states, install actions, positive feedback
- **Accent2 (#EC4899 - Pink)**: Delete actions, warnings, destructive operations
- **Background (#0F172A - Dark Slate)**: Main background color for dark mode

## How to Apply Changes

1. **Restart Streamlit**: The changes will take effect when you restart the Streamlit app
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Rebuild Frontend Components** (if needed):
   - For navbar component:
     ```bash
     cd streamlit_navbar/frontend
     npm run build
     ```
   - For markdown select component:
     ```bash
     cd streamlit_markdown_select/frontend
     npm run build
     ```

## Notes

- The color scheme is already defined in `bg.json` under the presentation theme
- All colors follow the same values specified in your request
- Dark mode is the primary theme for the application
- The new colors provide better contrast and a more modern, professional look
- Purple (#6B46C1) replaces the previous red (#FF4B4B) as the primary brand color
