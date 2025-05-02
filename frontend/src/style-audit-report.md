# Frontend Style Audit Report

## Overview
This document provides a comprehensive audit of all styling in the frontend codebase, identifying color values, font declarations, size values, border styles, shadows, and spacing tokens. This audit will serve as the foundation for creating a centralized theme.css file.

## CSS Variables
The project already uses CSS variables defined in `:root` selectors, which is a good foundation for centralization. However, there are inconsistencies between files and some hardcoded values that should be standardized.

### Color Values

#### Primary Colors
| Variable | Value | Used In | Notes |
|----------|-------|---------|-------|
| `--background-color` | `#2E2E38` | App.css, main.css | Main background color |
| `--primary-color` | `#1f1f1f` | App.css, main.css, multiple components | Primary UI element color |
| `--secondary-color` | `#1e1e1e` | App.css, main.css, multiple components | Secondary UI element color |
| `--accent-color` | `#bb86fc` | App.css, main.css, multiple components | Accent color for highlights and interactive elements |
| `--text-color` | `#ffffff` | App.css, main.css, multiple components | Primary text color |
| `--secondary-text-color` | `#bbbbbb` | App.css, main.css | Secondary text color |
| `--border-color` | `#333333` | App.css, main.css, multiple components | Border color for UI elements |

#### Button Colors
| Variable | Value | Used In | Notes |
|----------|-------|---------|-------|
| `--button-background` | `#bb86fc` | App.css, main.css | Default button background |
| `--button-hover` | `#985eff` | App.css, main.css | Button hover state |
| `--button-text-color` | `#ffffff` | App.css, main.css | Button text color |

#### Input and Form Colors
| Variable | Value | Used In | Notes |
|----------|-------|---------|-------|
| `--input-background` | `#2c2c2c` | App.css, main.css | Input field background |
| `--input-text-color` | `#ffffff` | App.css, main.css | Input field text color |
| `--dropzone-border` | `#444444` | App.css, main.css | Border color for file dropzone |
| `--dropzone-background` | `#1f1f1f` | App.css, main.css | Background for file dropzone |
| `--dropzone-active-background` | `#2c2c2c` | App.css, main.css | Active state for file dropzone |
| `--modal-background` | `#1f1f1f` | App.css, main.css | Modal background color |

#### Hardcoded Colors (TrainingTab.css)
| Color | Used For | Notes |
|-------|----------|-------|
| `#f8f9fa` | Background for panels, containers | Light background color |
| `#e9ecef` | Disabled input background | Light gray |
| `#28a745` | Start button background | Green |
| `#218838` | Start button hover | Darker green |
| `#8fd19e` | Disabled start button | Light green |
| `#dc3545` | Stop button background | Red |
| `#c82333` | Stop button hover | Darker red |
| `#e4868e` | Disabled stop button | Light red |
| `#f8d7da` | Error background | Light red |
| `#f5c6cb` | Error border | Pink |
| `#721c24` | Error text | Dark red |
| `#e0e0e0` | Borders, dividers | Light gray |
| `#fff8e1` | Warning background | Light yellow |
| `#ffecb3` | Warning border | Yellow |
| `#e3f2fd` | Label item background | Light blue |
| `#bbdefb` | Label item border | Blue |
| `#a5d6a7` | Completed step number | Light green |
| `#2e7d32` | Completed step number text | Dark green |
| `#90caf9` | Current step number | Light blue |
| `#1565c0` | Current step number text | Dark blue |
| `#ffc107` | Warning border | Yellow |
| `#2196f3` | Link button | Blue |
| `#1976d2` | Link button hover | Darker blue |
| `#ffebee` | Validation warning background | Light red |
| `#f44336` | Validation warning border | Red |
| `#343a40` | Logs container background | Dark gray |
| `#adb5bd` | Log message color | Gray |
| `#6c757d` | Log time color | Medium gray |

#### Chart Colors (MetricsVisualizer.js)
| Color | Used For | Notes |
|-------|----------|-------|
| `rgb(255, 99, 132)` | Training loss border | Red |
| `rgba(255, 99, 132, 0.1)` | Training loss background | Transparent red |
| `rgb(54, 162, 235)` | Validation loss border | Blue |
| `rgba(54, 162, 235, 0.1)` | Validation loss background | Transparent blue |
| `rgb(75, 192, 192)` | Training accuracy border | Teal |
| `rgba(75, 192, 192, 0.1)` | Training accuracy background | Transparent teal |
| `rgb(153, 102, 255)` | Validation accuracy border | Purple |
| `rgba(153, 102, 255, 0.1)` | Validation accuracy background | Transparent purple |
| `rgba(153, 102, 255, 0.6)` | Bar chart background | Semi-transparent purple |
| `rgba(153, 102, 255, 1)` | Bar chart border | Purple |

### Font Declarations

#### Font Families
| Value | Used In | Notes |
|-------|---------|-------|
| `'Segoe UI', Tahoma, Geneva, Verdana, sans-serif` | App.css (body) | Primary font family |
| `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif` | main.css (body, html), ConfigurationPanel.css | System font stack |
| `'Courier New', monospace` | TrainingTab.css (logs-container) | Monospace font for logs |
| `sans-serif` | index.css (body, html) | Generic sans-serif fallback |

#### Font Sizes
| Value | Used In | Notes |
|-------|---------|-------|
| `1.8em` | App.css (.app-name) | Large text |
| `1.5rem` | TrainingTab.css (h2) | Large heading |
| `1.3rem` | main.css (.info-modal-title), TrainingTab.css (h3) | Medium-large heading |
| `1.25rem` | TrainingTab.css (.metric-value) | Large value text |
| `1.2rem` | App.css (.sidebar h2) | Medium heading |
| `1.1rem` | App.css (.nav-link), TrainingTab.css (h3, h4) | Medium text |
| `1rem` | TrainingTab.css (process-button), ConfigurationPanel.css (process-button) | Default button text |
| `0.95rem` | main.css (.info-modal-description p) | Slightly smaller than default |
| `0.9rem` | Multiple locations | Small text |
| `0.85rem` | Multiple locations | Smaller text |
| `0.8rem` | ConfigurationPanel.css (.remove-feature-btn) | Very small text |

#### Font Weights
| Value | Used In | Notes |
|-------|---------|-------|
| `700` | App.css (.app-name), TrainingTab.css (.metric-value) | Bold |
| `600` | Multiple locations | Semi-bold |
| `500` | Multiple locations | Medium |
| `normal` | TrainingTab.css (.workflow-steps li) | Normal weight |

### Size Values

#### Spacing (Margins, Paddings)
| Value | Used In | Notes |
|-------|---------|-------|
| `30px` | App.css (.app-header padding, .logo-container left) | Large spacing |
| `20px` | Multiple locations | Standard content padding |
| `15px` | Multiple locations | Medium spacing |
| `12px` | Multiple locations | Medium-small spacing |
| `10px` | Multiple locations | Small spacing |
| `8px` | Multiple locations | Very small spacing |
| `5px` | Multiple locations | Minimal spacing |
| `1.5rem` | TrainingTab.css (multiple locations) | Large responsive spacing |
| `1.25rem` | TrainingTab.css (multiple locations) | Medium-large responsive spacing |
| `1rem` | TrainingTab.css (multiple locations) | Standard responsive spacing |
| `0.75rem` | TrainingTab.css (multiple locations) | Small responsive spacing |
| `0.5rem` | TrainingTab.css (multiple locations) | Very small responsive spacing |
| `0.25rem` | TrainingTab.css (multiple locations) | Minimal responsive spacing |

#### Element Sizes
| Value | Used In | Notes |
|-------|---------|-------|
| `400px` | App.css (.sidebar width), main.css (.react-flow-container height) | Large fixed size |
| `350px` | TrainingTab.css (.training-config-panel flex-basis) | Panel width |
| `300px` | TrainingTab.css (.chart-wrapper height) | Chart height |
| `250px` | main.css (.chart-container height) | Smaller chart height |
| `90px` | App.css (.app-header height) | Header height |
| `50px` | App.css (.app-logo height) | Logo height |
| `24px` | TrainingTab.css (.step-number width/height) | Circle step indicator |
| `16px` | App.css (checkbox width/height) | Checkbox size |
| `2px` | App.css (.nav-link::after height) | Thin underline |
| `1px` | Multiple locations | Standard border width |
| `100%` | Multiple locations | Full width/height |
| `80%` | App.css (.file-uploader width, .graph-section width) | Most of container width |
| `15%` | App.css (.reopen-flow-btn width) | Small portion of width |

### Border Styles
| Value | Used In | Notes |
|-------|---------|-------|
| `1px solid var(--border-color)` | Multiple locations | Standard border |
| `2px dashed var(--dropzone-border)` | App.css, main.css (.dropzone) | Dashed border for dropzone |
| `1px solid #e0e0e0` | TrainingTab.css (multiple elements) | Light gray border |
| `1px solid #ffecb3` | TrainingTab.css (.graph-summary.no-graph) | Yellow warning border |
| `1px solid #f5c6cb` | TrainingTab.css (.training-error) | Red error border |
| `1px solid #bbdefb` | TrainingTab.css (.label-item) | Blue label border |
| `border-left: 4px solid #ffc107` | TrainingTab.css (.workflow-message.warning) | Yellow left border |
| `border-left: 4px solid #f44336` | TrainingTab.css (.validation-warning) | Red left border |
| `border-left: 3px solid var(--accent-color)` | App.css (.sidebar-nav-link.active) | Purple active indicator |
| `border-radius: 8px` | Multiple locations | Large rounded corners |
| `border-radius: 6px` | Multiple locations | Medium-large rounded corners |
| `border-radius: 5px` | Multiple locations | Medium rounded corners |
| `border-radius: 4px` | Multiple locations | Standard rounded corners |
| `border-radius: 50%` | TrainingTab.css (.step-number) | Circle |

### Shadows
| Value | Used In | Notes |
|-------|---------|-------|
| `box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2)` | App.css (.app-header) | Medium shadow |
| `box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3)` | App.css (.app-header:hover) | Larger shadow on hover |
| `box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08)` | TrainingTab.css (panels) | Subtle shadow |
| `box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1)` | App.css (.config-section) | Small shadow |
| `box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1)` | main.css (.feature-summary) | Very subtle shadow |
| `box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15)` | main.css (.feature-summary:hover) | Enhanced shadow on hover |
| `box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2)` | main.css (.process-button) | Medium button shadow |
| `box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3)` | main.css (.process-button:hover) | Enhanced button shadow on hover |
| `box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.1)` | TrainingTab.css (.metrics-container) | Inset shadow |
| `box-shadow: 0 0 0 2px rgba(187, 134, 252, 0.25)` | main.css (input focus) | Focus ring |
| `box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2)` | ConfigurationPanel.css (input focus) | Alternative focus ring |

### Transitions
| Value | Used In | Notes |
|-------|---------|-------|
| `transition: color 0.3s ease` | App.css (.nav-link) | Color transition |
| `transition: width 0.3s ease` | App.css (.nav-link::after) | Width transition |
| `transition: background 0.3s` | Multiple locations | Background color transition |
| `transition: all 0.3s ease` | App.css (.sidebar-nav-link) | All properties transition |
| `transition: all 0.2s ease` | Multiple locations | Faster all properties transition |
| `transition: background-color 0.2s` | TrainingTab.css (buttons) | Background color transition |
| `transition: background-color 0.2s ease` | ConfigurationPanel.css (buttons) | Background color transition with easing |
| `transition: border-color 0.2s ease` | ConfigurationPanel.css (form inputs) | Border color transition |

## Inline Styles
The codebase has minimal inline styles, primarily in the Sidebar.js component:

| Component | Style | Used For |
|-----------|-------|----------|
| Sidebar.js | `style={{ marginRight: 8 }}` | Icon spacing for FiActivity, FiCpu, FiDatabase, FiDownload, FiBarChart2 |

## Recommendations for Centralization

1. **Create a comprehensive theme.css file** with all color variables, typography, spacing, and other design tokens.

2. **Standardize color palette**:
   - Consolidate similar colors (e.g., multiple shades of gray)
   - Create a systematic color naming convention (primary, secondary, success, danger, etc.)
   - Define color variants (light, dark) for each base color

3. **Create typography system**:
   - Standardize font sizes using a scale (xs, sm, md, lg, xl, etc.)
   - Define consistent font weights
   - Establish line heights and letter spacing

4. **Define spacing system**:
   - Create a spacing scale (xs, sm, md, lg, xl or numeric scale)
   - Replace hardcoded pixel values with spacing variables

5. **Standardize component styles**:
   - Create consistent styles for buttons, inputs, cards, etc.
   - Define states (hover, active, disabled) for interactive elements

6. **Remove duplicate styles** between App.css and main.css

7. **Replace hardcoded values** in TrainingTab.css with variables from the centralized theme

8. **Convert inline styles** in Sidebar.js to use CSS classes with appropriate spacing variables

This audit provides a comprehensive overview of the current styling approach and will serve as the foundation for creating a centralized theme.css file in the next step.