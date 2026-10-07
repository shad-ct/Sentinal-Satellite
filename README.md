# Sentinel-2 Satellite Image Change Detection System 🛰️🌍

A local-first, full-stack satellite imagery change detection application designed to automatically acquire, process, and analyze **ESA Copernicus Sentinel-2** imagery to detect meaningful geographical and structural changes over time.

---

## 📖 Project Overview & Theoretical Foundation

Earth Observation (EO) data is highly complex. A single satellite image from the Sentinel-2 constellation is not just a standard RGB photograph; it is a multi-dimensional array of data capturing light across 13 different spectral bands (ranging from visible light to near-infrared and short-wave infrared). 

This project aims to harness this multi-spectral data to automatically detect **meaningful structural changes** in the landscape—such as urban sprawl, deforestation, agricultural shifts, and water body shrinkage—while aggressively filtering out transient variations like cloud cover and minor seasonal shadows.

The project is structured into iterative phases. The current phase implements a highly reliable **Classical Computer Vision Baseline**, which serves as the foundational data acquisition pipeline and ground-truth generator. The subsequent phases will replace the classical differencing engine with deep learning architectures (Siamese U-Nets and Transformers).

---

## 🛠️ Technology Stack & Tooling

The architecture is designed to be highly modular, separating the heavy geospatial processing from the client-side visualization.

*   **Backend / Server**: `FastAPI` (Python)
    *   Provides high-performance, asynchronous endpoints for UI interactions and background task management.
*   **Geospatial & Data Processing (Python)**:
    *   `rasterio` & `gdal`: Used for reading, writing, and projecting geographic data formats (GeoTIFF, JPEG2000). They handle the affine transformations mapping pixels to exact GPS coordinates.
    *   `numpy`: The core mathematical engine. Satellite bands are loaded as multi-dimensional numpy arrays (matrices) for extremely fast element-wise arithmetic (critical for calculating spectral indices across millions of pixels).
    *   `scipy` & `scikit-image` (`skimage`): Utilized for morphological operations, structural noise removal, and Gaussian smoothing on the change masks.
    *   `shapely` & `pyproj`: For parsing user bounding boxes, converting coordinate reference systems (CRS), and defining geometric areas.
*   **Database**: `SQLite` via `SQLAlchemy` 
    *   Acts as a local cache registry. Instead of repeatedly downloading large satellite tiles, the database tracks local observation files, their exact geospatial boundaries, and cloud cover metrics.
*   **Frontend**: `Vanilla JavaScript`, `HTML5`, `TailwindCSS/CSS`, and `Leaflet.js`
    *   Provides a highly responsive, modern "glassmorphism" web dashboard. 
    *   `Leaflet.js` is paired with CARTO Dark Matter basemaps to overlay our generated change detection heatmaps perfectly onto global maps.

---

## 📡 Remote Sensing Architecture: APIs & Data Acquisition

Acquiring analysis-ready data is notoriously difficult in remote sensing. We directly interface with the **Copernicus Data Space Ecosystem (CDSE)** via dual strategies to guarantee data acquisition.

### 1. Catalog Search (OData API)
When a user defines a bounding box, the backend builds a spatial query against the CDSE OData API. It searches the catalogue for **Sentinel-2 Level-2A (L2A)** products. 
*   *Why Level-2A?* L2A data is already **atmospherically corrected** (Bottom of Atmosphere reflectance). This means atmospheric scattering (haze, aerosols) has been algorithmically removed, providing true surface reflectance values required for accurate temporal comparison.

### 2. Sentinel Hub Process API (Primary Engine)
Downloading a full Sentinel-2 `.SAFE` archive involves pulling 800MB+ of data covering a 100x100km tile, even if the user only wants a 5x5km area.
To solve this, we utilize the **Sentinel Hub Process API**. The backend dynamically writes a server-side JavaScript **evalscript** and POSTs it to the CDSE servers. This script instructs the server to:
1. Crop the data to the exact user bounding box.
2. Extract only the specific multi-spectral bands required for our algorithms (B02, B03, B04, B08, B11, B12).
3. Extract the `SCL` (Scene Classification Layer) for cloud masking.
4. Reproject the data and stream it back directly as a lightweight, analysis-ready **Float32 Cloud-Optimized GeoTIFF**.

### 3. CDSE S3 / OData Streaming (Fallback Engine)
If the Process API fails due to rate limits or backend timeouts, our system engages an automatic fallback. It authenticates with the CDSE Keycloak service to acquire an OAuth2 token, handles complex geographic HTTP redirects, and directly streams the raw `.SAFE` product zip file from CDSE's internal AWS S3 buckets. It then manually parses the XML manifests, extracts the required JPEG2000 bands, and aligns them locally using `rasterio`.

---

## 🔬 Theoretical Implementation: How Changes are Calculated

The current implementation uses a **Classical Computer Vision Differencing Engine**. It does not simply subtract RGB pixels; it evaluates the underlying physical properties of the earth's surface using normalized indices.

### 1. Spectral Index Theory
Different materials absorb and reflect light at specific wavelengths. By analyzing the ratio between specific spectral bands, we can isolate physical features:
*   **NDVI (Normalized Difference Vegetation Index)**: `(NIR - RED) / (NIR + RED)`. Healthy vegetation strongly absorbs Red light (for photosynthesis) and strongly reflects Near-Infrared (NIR) light. NDVI values range from -1 to 1, perfectly isolating live biomass.
*   **NDWI (Normalized Difference Water Index)**: `(GREEN - NIR) / (GREEN + NIR)`. Water strongly absorbs NIR, allowing us to accurately map shorelines, floods, and droughts.
*   **NDBI (Normalized Difference Built-up Index)**: `(SWIR - NIR) / (SWIR + NIR)`. Short-Wave Infrared (SWIR) is highly reflected by urban surfaces (concrete, asphalt). NDBI highlights structural development.

### 2. The Differencing Engine Workflow
To detect change between $T_0$ (Before) and $T_1$ (After):
1.  **Cloud Masking**: The algorithm reads the Sentinel-2 `SCL` band. Any pixel classified as cloud (high probability), cloud shadow, or defective is mathematically masked out (set to `NaN` or zero weight) in both images to prevent false change detection.
2.  **Index Differencing**: The indices (NDVI, NDWI, NDBI) are computed for both $T_0$ and $T_1$. The absolute temporal difference is calculated: $\Delta Index = |Index_{T_1} - Index_{T_0}|$.
3.  **Aggregation & Gaussian Smoothing**: The differences are aggregated into a composite "Change Score". A continuous Gaussian blur ($\sigma = 1.0$) is applied via `scikit-image`. *Theory:* Satellite orbits are not perfect; images often suffer from 1-2 pixel micro-misalignments (registration errors) and sensor noise. Gaussian smoothing acts as a low-pass filter, neutralizing high-frequency spatial noise.
4.  **Thresholding**: A binary threshold (configurable via the UI) isolates statistically significant changes (e.g., clearing a forest creates a massive shift in NDVI) from minor variations (e.g., a lawn drying out slightly).
5.  **Morphological Cleanup**: Using morphological `closing` and `opening` operations, the algorithm removes isolated "salt and pepper" noise pixels and fills tiny holes inside actual change zones, ensuring the output mask represents contiguous, meaningful regions.

---

## 🤖 Future AI Inference & Capabilities (Deep Learning)

While the classical baseline is highly effective for massive, obvious changes, it hits theoretical limits when dealing with complex **seasonal variations** (e.g., differentiating between a forest changing colors in Autumn vs. a forest being actively deforested). To solve this, the pipeline is designed to feed into Deep Learning architectures.

### 1. Siamese U-Nets (Planned Phase)
A Siamese U-Net is a convolutional neural network (CNN) architecture designed specifically for change detection. 
*   **Theory**: It consists of two identical "Encoder" networks that share the same weights. $T_0$ is fed into Encoder A, and $T_1$ into Encoder B. The encoders extract deep spatial features (edges, textures, semantic context). The feature maps are then concatenated and passed through a "Decoder" network that upsamples the data back to the original resolution, outputting a highly accurate binary change mask. 
*   **Advantage**: Because it learns spatial *context*, it can ignore seasons and focus on structural reality.

### 2. ChangeFormer (Transformers)
A state-of-the-art vision transformer architecture.
*   **Theory**: Instead of relying solely on local convolutions, it uses *Self-Attention Mechanisms* to understand the global context of the entire landscape. It learns relationships between distant pixels, allowing for unprecedented accuracy in complex terrains.

### What inferences can we build out of this?
With these AI models attached to our automated data pipeline, the system can provide powerful inferences:
*   **Deforestation Tracking**: Automatically alerting authorities to illegal logging by classifying vegetative loss.
*   **Urban Expansion Monitoring**: Helping city planners track uncontrolled suburban sprawl by mapping new concrete structures over time.
*   **Disaster Assessment**: Comparing pre- and post-flood/earthquake imagery to automatically map destroyed infrastructure and guide rescue efforts.
*   **Agricultural Yield Prediction**: Monitoring crop cycles, soil moisture, and harvest patterns across vast farmlands without human intervention.

---

## 🚀 Installation & Running

Follow these steps to set up the system locally.

### 1. Clone the Repository
```bash
git clone https://github.com/shad-ct/Sentinal-Satellite.git
cd Sentinal-Satellite
```

### 2. Setup the Python Environment
Ensure you have Python 3.10+ installed.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API Keys Configuration
You need two API keys for this project to work fully:

**A. Copernicus Data Space Ecosystem (CDSE) Key**
*   **Where to find**: Register for a free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/). Go to your Dashboard -> API Keys and generate a new OAuth client.
*   **Where to put**: Create a `.env` file in the root directory of the project and add your credentials:
    ```ini
    COPERNICUS_CLIENT_ID=your_client_id_here
    COPERNICUS_CLIENT_SECRET=your_client_secret_here
    ```

**B. CARTO Basemaps API Key (For the Web UI Map)**
*   **Where to find**: Go to [carto.com/basemaps/apikey](https://carto.com/basemaps/apikey/) to get a free API key for rendering the dark matter map tiles.
*   **Where to put**: Open `backend/templates/index.html` and replace the `key=` parameter in the `L.tileLayer` URL (around line 713) with your actual API key.

### 4. Running the Web Dashboard
Start the FastAPI server:
```bash
uvicorn backend.app.main:app --reload
```
Then, open your web browser and navigate to **[http://localhost:8000](http://localhost:8000)**.

### 5. Running via Command Line (Optional)
If you prefer to bypass the UI, you can use the CLI scripts:
```bash
# Download a Sentinel-2 image for a specific location and date
python scripts/download_image.py --lat 11.8745 --lon 75.3704 --date 2024-03-15 --size-km 10 --cloud 20

# Run the Change Detection algorithm between two dates
python scripts/compare_images.py --lat 11.8745 --lon 75.3704 --before 2024-02-15 --after 2024-03-15
```
