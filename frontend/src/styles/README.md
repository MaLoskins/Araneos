# Frontend Styling Guide

## Overview

This project uses a centralized theming approach with CSS variables to maintain consistent styling across the application. The main styling files are:

- `theme.css`: The core styling file that defines all variables and base styles
- `main.css`: Component-specific styles that use variables from theme.css
- Component-specific CSS files: For styles that only apply to specific components

## Theme Structure

The theme is organized into the following sections:

### Color System

Colors are defined in a two-tier system:
1. Base color definitions (e.g., `--color-primary`, `--color-dark`)
2. Semantic assignments (e.g., `--background-color`, `--text-color`)

This approach allows changing the entire color scheme by modifying just the semantic assignments.

### Typography

Font families, sizes, weights, and line heights are defined as variables:
- `--font-primary`, `--font-secondary`, `--font-mono`
- `--font-size-xs` through `--font-size-4xl`
- `--font-weight-normal`, `--font-weight-medium`, etc.

### Spacing

Two spacing systems are available:
- Relative spacing: `--space-xs` through `--space-3xl` (using rem units)
- Fixed spacing: `--space-4` through `--space-40` (using px units)

### Borders, Shadows, and Transitions

- Border widths and radii
- Box shadows for different elevation levels
- Transition durations and timing functions

## How to Use

### Using Theme Variables

Always use theme variables instead of hardcoded values:

```css
/* ❌ Don't do this */
.my-component {
  color: #bb86fc;
  padding: 10px;
  font-size: 16px;
}

/* ✅ Do this instead */
.my-component {
  color: var(--accent-color);
  padding: var(--space-sm);
  font-size: var(--font-size-base);
}
```

### Adding New Components

1. First, try to use existing utility classes from theme.css
2. If component-specific styles are needed, create a dedicated CSS file
3. Import the CSS file in your component
4. Use theme variables for all values

Example:

```jsx
// MyComponent.js
import React from 'react';
import './MyComponent.css';

const MyComponent = () => (
  <div className="my-component">
    <h2>My Component</h2>
    <p>Content goes here</p>
  </div>
);

export default MyComponent;
```

```css
/* MyComponent.css */
.my-component {
  background-color: var(--primary-color);
  padding: var(--space-lg);
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-md);
}

.my-component h2 {
  font-size: var(--font-size-xl);
  margin-bottom: var(--space-sm);
}
```

### Responsive Design

The theme includes media queries for responsive design. Use them to adjust styles for different screen sizes:

```css
@media (max-width: 768px) {
  .my-component {
    padding: var(--space-md);
  }
}
```

## Utility Classes

Theme.css provides utility classes for common styling needs:

- Text alignment: `.text-center`, `.text-left`, `.text-right`
- Font weights: `.font-normal`, `.font-medium`, `.font-semibold`, `.font-bold`
- Display: `.flex`, `.block`, `.inline-block`, `.grid`
- Flex utilities: `.flex-row`, `.flex-col`, `.items-center`, etc.
- Spacing: `.m-0`, `.p-0`
- Width/Height: `.w-full`, `.h-full`
- Overflow: `.overflow-hidden`, `.overflow-auto`, etc.
- Position: `.relative`, `.absolute`, `.fixed`
- Borders: `.rounded-sm` through `.rounded-full`
- Shadows: `.shadow-xs` through `.shadow-2xl`

## Maintaining the Theme

When adding new styles:

1. Check if a variable already exists for your use case
2. If not, add new variables to theme.css in the appropriate section
3. Follow the naming conventions established in the theme
4. Document any new variables or utility classes in this README

## Best Practices

- Always use semantic variable names (e.g., `--button-background` instead of `--blue`)
- Maintain the separation between base colors and semantic assignments
- Use utility classes for common styling needs
- Create component-specific CSS files for complex components
- Keep component-specific styles minimal by leveraging the theme
- Test all styles on different screen sizes