# CSS Centralization Migration Guide

## Overview

This document outlines the steps to migrate the existing CSS to the new centralized theming system. The migration has been started with the creation of `theme.css` and updating a few key files, but there is still work to be done to fully implement the centralized approach across the entire codebase.

## Completed Changes

1. **Style Audit**: A comprehensive audit of all styling in the frontend codebase has been completed and documented in `style-audit-report.md`.

2. **Centralized Theme**: Created `theme.css` with all color variables, typography, spacing, and other design tokens.

3. **Updated Core Files**:
   - Modified `index.js` to import the new theme.css file
   - Created `Sidebar.css` and updated `Sidebar.js` to use CSS classes instead of inline styles

4. **Documentation**:
   - Created a styling guide in `styles/README.md`
   - Created this migration guide

## Remaining Tasks

The following tasks need to be completed to fully migrate to the centralized theme:

### 1. Update CSS Files

Replace hardcoded values with theme variables in these files:

- [ ] `App.css`
- [ ] `main.css`
- [ ] `TrainingTab.css`
- [ ] `ConfigurationPanel.css`

For each file:
1. Identify hardcoded values (colors, spacing, font sizes, etc.)
2. Replace them with the corresponding theme variables
3. Remove duplicate styles that are already defined in theme.css

### 2. Component-Specific CSS

For components with inline styles or component-specific styling needs:

- [ ] Create dedicated CSS files for components that need them
- [ ] Import these CSS files in the corresponding component files
- [ ] Replace inline styles with CSS classes
- [ ] Use theme variables for all values

Priority components to address:
- [ ] InfoButton.js
- [ ] InfoModal.js
- [ ] GraphNet.js and related components
- [ ] TrainingTab.js and related components

### 3. Consolidate Duplicate Styles

- [ ] Identify and remove duplicate styles between App.css and main.css
- [ ] Move common component styles to main.css
- [ ] Keep only component-specific styles in component CSS files

### 4. Testing

After making changes to each file:

- [ ] Test the UI to ensure styles are applied correctly
- [ ] Test on different screen sizes to ensure responsive design works
- [ ] Verify that no styling regressions have been introduced

## Migration Strategy

### Approach

We recommend a gradual, component-by-component approach:

1. Start with smaller, simpler components
2. Move to larger, more complex components
3. Finally, update the main layout and global styles

### Step-by-Step Process for Each Component

1. **Analyze the component**:
   - Identify all styles used by the component
   - Note any inline styles or hardcoded values

2. **Create a component-specific CSS file** (if needed):
   ```css
   /* ComponentName.css */
   .component-name {
     /* Use theme variables */
     color: var(--text-color);
     padding: var(--space-md);
   }
   ```

3. **Import the CSS file** in the component:
   ```jsx
   import './ComponentName.css';
   ```

4. **Replace inline styles** with CSS classes:
   ```jsx
   // Before
   <div style={{ marginRight: 8 }}>

   // After
   <div className="component-name-item">
   ```

5. **Test the component** to ensure styles are applied correctly

### Example: Converting a Component

**Before**:
```jsx
// MyComponent.js
const MyComponent = () => (
  <div style={{ backgroundColor: '#1f1f1f', padding: '20px' }}>
    <h2 style={{ color: '#bb86fc', fontSize: '1.2rem' }}>Title</h2>
    <p style={{ color: '#ffffff' }}>Content</p>
  </div>
);
```

**After**:
```jsx
// MyComponent.js
import './MyComponent.css';

const MyComponent = () => (
  <div className="my-component">
    <h2 className="my-component-title">Title</h2>
    <p>Content</p>
  </div>
);
```

```css
/* MyComponent.css */
.my-component {
  background-color: var(--primary-color);
  padding: var(--space-lg);
}

.my-component-title {
  color: var(--accent-color);
  font-size: var(--font-size-xl);
}

/* No need to specify text color as it inherits from theme.css */
```

## Best Practices During Migration

1. **One component at a time**: Complete the migration for one component before moving to the next
2. **Commit frequently**: Make small, focused commits for each component
3. **Test thoroughly**: Verify that styles are applied correctly after each change
4. **Use theme variables**: Always use theme variables instead of hardcoded values
5. **Document changes**: Update documentation as needed
6. **Maintain consistency**: Follow the naming conventions established in theme.css

## Timeline

Estimated time to complete the migration:

- Small components: 30 minutes per component
- Medium components: 1-2 hours per component
- Large components: 2-4 hours per component
- Global styles: 4-6 hours

Total estimated time: 20-30 hours depending on the complexity of the codebase.

## Support

If you have questions or need help with the migration, please refer to:

- The style audit report (`style-audit-report.md`)
- The styling guide (`styles/README.md`)
- This migration guide

## Conclusion

Migrating to a centralized theming system will improve maintainability, consistency, and developer experience. While the migration requires an upfront investment of time, it will pay off in the long run by making it easier to update styles, maintain consistency, and onboard new developers.