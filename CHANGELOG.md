# Changelog

All notable changes to the HOLE project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-03

### Added
- Initial release of HOLE library
- Core persistent homology computation using GUDHI
- Multiple distance metrics (Euclidean, cosine, Manhattan, Mahalanobis, geodesic)
- Comprehensive visualization suite:
  - Persistence diagrams and barcodes
  - Dimensionality reduction (PCA, t-SNE, MDS)
  - Cluster flow analysis with Sankey diagrams
  - Heatmaps and dendrograms
  - Scatter plots with convex hulls
- Main classes: `HOLEVisualizer`, `MSTProcessor`, `ClusterFlowAnalyzer`
- Simple examples for easy onboarding
- Comprehensive test suite

### Fixed
- **MAJOR**: Added examples

### Changed
- Improved error handling throughout codebase
- Enhanced logging system with proper log levels
- Cleaned up class naming conventions
- Reorganized visualization module structure
- Updated documentation to match actual API

### Technical Improvements
- Added type hints and comprehensive docstrings
- Implemented proper input validation for all main classes
- Structured logging instead of print statements
- Consistent exception handling patterns
- Modular, maintainable code architecture

### Repository Structure
- Clean examples directory with progressive complexity
- Proper .gitignore for Python projects
- Comprehensive README with usage examples
- Organized documentation structure
- Working test suite with good coverage

## Previous Versions

### [Pre-0.1.0] - Historical
- Initial development version with significant issues
- Broken test suite and API inconsistencies  
- Poor error handling and documentation
- Complex examples without simple onboarding path

---

**Legend:**
- **MAJOR**: Breaking changes or critical fixes
- **Added**: New features
- **Changed**: Changes in existing functionality  
- **Fixed**: Bug fixes
- **Removed**: Removed features
- **Security**: Security improvements
