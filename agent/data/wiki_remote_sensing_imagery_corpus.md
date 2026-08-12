Remote Sensing & Image Interpretation Knowledge Base

# Remote Sensing & Image Interpretation Knowledge Base  
*Generated from Wikipedia categories: Category:Remote sensing, Category:Satellite imagery, Category:Aerial photography, Category:Image processing, Category:Remote sensing software, Category:Earth observation satellites, Category:Photogrammetry, Category:Geographic information systems*

Remote Sensing & Image Interpretation Knowledge Base | Satellite crop monitoring | Overview

## Satellite crop monitoring  
### Overview  
Satellite crop monitoring is the technology which facilitates real-time crop vegetation index monitoring via spectral analysis of high resolution satellite images for different fields and crops which enables to track positive and negative dynamics of crop development. The difference in vegetation index informs about single-crop development disproportions that speaks for the necessity of additional agriculture works on particular field zones—that is because satellite crop monitoring belongs to precision agriculture methods.
Satellite crop monitoring technology allows to perform online crop monitoring on different fields, located in different areas, regions, even countries and on different continents. The technology's advantage is a high automation level of sown area condition and its interpretation in an interactive map which can be read by different groups of users.
Satellite crop monitoring technology users are:  
agronomists and agriculture companies management (crop vegetation control, crop yield forecasting, management decisions optimization);
business owners (business prospects estimates, making reasonable decisions on capital investments, providing information for management decisions);
investors and investment analysts (investment potential estimation, making investment decisions, making sustainable forecasts);
agriculture consultants and crop advisors (providing remote-sensing-based advisory services; integrating satellite imagery, vegetation indices, weather analytics, and scouting tools to optimize client field health, resource use, yield potential, and consulting reach);
insurance brokers (data collection, clients claims verification, scale of rates and insurance premium amounts calculation);
agriculture machinery producers (integration of crop monitoring solutions with agriculture machinery board computers operations, functional development);
state and sectoral organisations engaged in agriculture, food security and ecological problems.  
Crop Health Monitoring - e.g. color, size
Satellite crop monitoring is the technology which facilitates real-time crop [vegetation index] monitoring via [Spectroscopy|spectral analysis] of high resolution [satellite] images for different fields and crops which enables to track positive and negative dynamics of crop development. The difference in vegetation index informs about single-crop development disproportions that speaks for the necessity of additional agriculture works on particular field zones—that is because satellite crop monitoring belongs to [precision agriculture] methods.  
Satellite crop monitoring technology allows to perform online crop monitoring on different fields, located in different areas, regions, even countries and on different continents. The technology's advantage is a high automation level of sown area condition and its interpretation in an interactive map which can be read by different groups of users.  
Satellite crop monitoring technology users are:
* agronomists and agriculture companies management (crop vegetation control, crop yield forecasting, management decisions optimization);
* business owners (business prospects estimates, making reasonable decisions on capital investments, providing information for management decisions);
* investors and investment analysts (investment potential estimation, making investment decisions, making sustainable forecasts);
* agriculture consultants and crop advisors (providing remote-sensing-based advisory services; integrating satellite imagery, vegetation indices, weather analytics, and scouting tools to optimize client field health, resource use, yield potential, and consulting reach);
* insurance brokers (data collection, clients claims verification, scale of rates and insurance premium amounts calculation);
* agriculture machinery producers (integration of crop monitoring solutions with agriculture machinery board computers operations, functional development);
* state and sectoral organisations engaged in agriculture, [food security] and ecological problems.

Remote Sensing & Image Interpretation Knowledge Base | Satellite crop monitoring | See also

### See also
* [Normalized Difference Vegetation Index]
* [Precision agriculture]
* [Remote sensing]
* [Satellite imaging]

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Overview

## Class activation mapping  
### Overview  
Class activation mapping methods are explainable AI (XAI) techniques used to visualize the regions of an input image that are the most relevant for a particular task, especially image classification, in convolutional neural networks (CNNs). These methods generate heatmaps by weighting the feature maps from a convolutional layer according to their relevance to the target class.
CNNs are designed to process spatially structured data, such as images, exploiting a series of convolution, non-linear activation and pooling operations to extract relevant features contained in the so-called feature maps from input data. CNNs have proven to be highly effective in a variety of computer vision and image processing tasks. Despite their strengths, CNNs (and deep learning models more broadly) are described as black boxes due to their complex and non-transparent internal layers of representation. The need to interpret decision-making processes gave birth to XAI techniques.
Class activation mapping methods were originally developed for class-discriminative scenarios to visualize which parts of the input image influenced the classification decision, namely to visually highlight the regions of those feature maps that contribute most strongly to the prediction of a given class. More advanced versions of these methods are not limited to image classification tasks, but have been extended also to several vision-related tasks, such as object detection, image captioning, visual question answering, image segmentation, and medical image interpretation.  
Class activation mapping methods are [Explainable artificial intelligence|explainable AI (XAI)] techniques used to visualize the regions of an input image that are the most relevant for a particular task, especially [image classification], in [Convolutional neural network|convolutional neural networks (CNNs)]. These methods generate [Heat map|heatmaps] by weighting the feature maps from a convolutional layer according to their relevance to the target class. CNNs have proven to be highly effective in a variety of [computer vision] and [Digital image processing|image processing] tasks. Despite their strengths, CNNs (and deep learning models more broadly) are described as [black box]es due to their complex and non-transparent internal layers of representation. The need to interpret decision-making processes gave birth to XAI techniques. image captioning, visual question answering, image segmentation,.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Background

### Background  
The following methods laid the groundwork for the class activation maps approaches, forming the conceptual basis of using gradients to highlight class-discriminative regions.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Background | Class model visualization and saliency maps for convolutional neural networks

#### Class model visualization and saliency maps for convolutional neural networks  
The class model visualization and image-specific saliency maps approaches have been presented in the foundational work "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" by Karen Simonyan, Andrea Vedaldi, and [Andrew Zisserman]
and it generalizes the deconvnet method by Zeiler and Fergus.  
* Class model visualization synthesizes an artificial input image that strongly activates the output neurons associated with a target class. Given a trained, fixed model, this method starts with a zero-initialized image, backpropagates the gradients from the class score to the image pixels, updates the image pixels increasing the specific class scores and it repeats the pixel updating process, showing an encoded (idealized version) prototype of the class of interest.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Background | Guided backpropagation

#### Guided backpropagation  
The concept of guided backpropagation can be traced for the first time in the paper by Springenberg et al. "Striving For Simplicity: The All Convolutional Net" and also this method builds upon the work by Zeiler and Fergus "Visualizing and Understanding Convolutional Networks".

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Base versions

### Base versions
Key architectural network differences between CAM and Grad-CAM techniques, with visual example.
Class activation mapping and gradient-weighted class activation mapping are the original and most widely used methods for visual explanations in convolutional neural networks. These methods serve as the foundation for many later developments in explainable AI.  
Notation: In this article, the symbols i and j represent integer indices that disappear inside sums or averages, while x and y are the continuous (or up-sampled integer) coordinates of the final heat-map that is plotted.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Base versions | Class activation mapping (CAM)

#### Class activation mapping (CAM)  
Leonardo DiCaprio's suit CAM visual localization. Using a modified ResNet-50 backbone, the CAM-based localization model has been tasked with identifying Leonardo DiCaprio's suit, assigning that class a confidence score of 59.53%.  
Class activation mapping (CAM) was the first, and the original, version of CAM methods, and it gave the name to the whole category. The approach was firstly introduced by Zhou et al. in their seminal work "Learning Deep Features for Discriminative Localization".

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Base versions | Class activation mapping (CAM) | Global average pooling (GAP)

##### Global average pooling (GAP)
Global average pooling (GAP) represents the key element in the original CAM approach.  
It is a dimensionality reduction technique and, similarly to other pooling layers, it allows the downsampling of the feature maps, calculating representative values for a specific region of the feature map. The particularity of GAP is that it calculates a single value for an entire feature map, significantly reducing the model dimensions.. Grad-CAM computes the [gradient] of a target class score, the pre-softmax logit, with respect to the feature maps of a convolutional neural network. The gradients are global-average-pooled to obtain importance weights, which are used to compute a class-specific localization map by linearly weighting the feature maps. The result is a heatmap that highlights the regions in the input image that are the most influential for predicting the target class.  
The main advantage of Grad-CAM, with respect to the standard CAM, is that it is model agnostic (provided that the network still needs to be [Differentiable function|differentiable]), meaning that it generates visual explanation for any CNN-based network without architectural changes or re-training, making it broadly applicable to pre-trained models.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Base versions | Class activation mapping (CAM) | Mathematical description

##### Mathematical description  
Considering:
* y ^C  the logits (i.e. the pre-softmax activated neurons responsible for a certain class prediction) of interest;
* A ^k  the feature activated map for a specific convolutional layer;
* L ^C_{\text{Grad-CAM}}  ∈  \mathbb{R}^{u \times v}  the class-discriminative localization map, of width u and height v for any class c;
Grad-CAM, employing backpropagation, computes the logit gradient with respect to the feature map A ^k  as  
\frac{\partial{y^C}}{\partial{A^k}(i,j)}  
highlighting the importance of a certain class discrimination decision process of the logit.<br />
These gradients are global-average-pooled over each element of the feature map (hence, highlighting the "importance" of the elements of a feature map k for a target class C):  
\alpha_k^C = \frac{1}{uv} \sum_{i} \sum_{j} \frac{\partial{y^C}}{\partial{A^k}(i,j)}  
So, to account for the total number of feature maps, each of them is multiplied by its weight (via dot-product) and element-wise summation is done:  
\sum_k \alpha_k^C A^k  
It can be observed that, due to the intrinsic nature of the gradient operation, some elements of the weighted feature map will have negative value, so, since only elements that have increased the logit of the predicted class are of interest, a ReLU activation function is applied:  
L_{Grad-CAM}^C(x,y) = ReLU(\sum_k \alpha_k^C A^k(x,y))  
Lastly, the output heatmap image dimensions are [Upsampling|upsampled] to the original image size to match the input dimensions.  
Notation: (a,b) indexes all pixel positions in the feature‐map, exactly like (i,j) does, but for the summation in the denominator.

Remote Sensing & Image Interpretation Knowledge Base | Class activation mapping | Base versions | Score-CAM

#### Score-CAM  
Score-CAM is a gradient-free CAM technique, thus redefining the original Grad-CAM and Grad-CAM++ working principles. It uses the model confidence scores instead of gradients.  
Score-CAM performs the following operations:

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Digital image correlation and tracking | Deformation mapping

### Deformation mapping  
For deformation mapping, the mapping function that relates the images can be derived from comparing a set of subwindow pairs over the whole images. (Figure 1). The coordinates or grid points (xi, yj) and (xi*, yj*) are related by the translations that occur between the two images. If the deformation is small and perpendicular to the [optical axis] of the camera, then the relation between (xi, yj) and (xi*, yj*) can be approximated by a 2D [affine transformation] such as:  
: x^* = x + u + \frac{\partial u}{\partial x}\Delta x + \frac{\partial u}{\partial y}\Delta y,
: y^* = y + v + \frac{\partial v}{\partial x}\Delta x + \frac{\partial v}{\partial y}\Delta y.  
Here u and v are translations of the center of the sub-image in the X and Y directions respectively. The distances from the center of the sub-image to the point (x, y) are denoted by  \Delta x  and  \Delta y . Thus, the correlation coefficient rij is a function of displacement components (u, v) and displacement gradients
: \frac{\partial u}{\partial x},\frac{\partial u}{\partial y},\frac{\partial v}{\partial x},\frac{\partial v}{\partial y}.  
Basic concept of deformation mapping by DIC  
DIC has proven to be very effective at mapping deformation in macroscopic mechanical testing, where the application of specular markers (e.g. paint, toner powder) or surface finishes from machining and polishing provide the needed contrast to correlate images well. However, these methods for applying surface contrast do not extend to the application of free-standing thin films for several reasons. First, vapor deposition at normal temperatures on semiconductor grade substrates results in mirror-finish quality films with [Root mean square|RMS] roughnesses that are typically on the order of several nanometers. No subsequent polishing or finishing steps are required, and unless electron imaging techniques are employed that can resolve microstructural features, the films do not possess enough useful surface contrast to adequately correlate images. Typically this challenge can be circumvented by applying paint that results in a random [speckle pattern] on the surface, although the large and turbulent forces resulting from either spraying or applying paint to the surface of a free-standing [thin film] are too high and would break the specimens. In addition, the sizes of individual paint particles are on the order of μms, while the film thickness is only several hundred nanometers, which would be analogous to supporting a large boulder on a thin sheet of paper.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Water remote sensing | Overview

## Water remote sensing  
### Overview  
Water Remote Sensing is the observation of water bodies such as lakes, oceans, and rivers from a distance in order to describe their color, state of ecosystem health, and productivity. Water remote sensing studies the color of water through the observation of the spectrum of water leaving radiance. From the spectrum of color coming from the water, the concentration of optically active components of the upper layer of the water body can be estimated via specific algorithms.
Water quality monitoring by remote sensing and close-range instruments has obtained considerable attention since the founding of EU Water Framework Directive.  
Water Remote Sensing is the observation of water bodies such as [lakes], [oceans], and [rivers] from a distance in order to describe their color, state of ecosystem health, and productivity. Water remote sensing studies the [color of water] through the observation of the [spectrum] of water leaving radiance. From the spectrum of color coming from the water, the concentration of optically active components of the upper layer of the water body can be estimated via specific [algorithms].
[Water quality] monitoring by [remote sensing] and close-range instruments has obtained considerable attention since the founding of EU [Water Framework Directive]. Thus, the values of remote sensing reflectance, an AOP, will change with changes in the optical properties and concentrations of the optically active substances in the water. Properties and concentrations of substances in the water are known as the inherent optical properties or IOPs. However, the development of water remote sensing techniques (by the use of satellite imaging, aircraft or close range optical devices) didn't start until the early 1970s. These first techniques measured the [Electromagnetic spectrum|spectral] and thermal differences in the emitted energy from water surfaces. In general, empirical relationships were settled between the spectral properties and the water quality parameters of the water body. In 1974, Ritchie et al. (1974)  developed an empirical approach to determine suspended sediments. This kind of empirical models are only able to use to determine water quality parameters of water bodies with similar conditions. In 1992 an analytical approach was used by Schiebe et al. (1992). This approach was based on the optical characteristics of water and water quality parameters to elaborate a physically based model of the relationship between the spectral and physical properties of the surface water studied. This physically based model was successfully applied in order to estimate suspended sediment concentrations.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Water remote sensing | Applications

### Applications
Example of specific phytoplankton absorption spectra. In this graph the characteristic blue and red Ch-a peaks at 438 nm and 676 nm can be seen. Another visible peak is the Cyanophicocianin absorption maximum at 624 nm.By the use of optical close range devices (e.g. [spectrometers], [radiometers]), airplanes or helicopters (airborne remote sensing) and satellites (space-borne remote sensing), the light energy radiating from water bodies is measured. For instance, algorithms are used to retrieve parameters such as [chlorophyll-a](Chl-a) and Suspended Particulate Matter (SPM) concentration, the absorption by [colored dissolved organic matter] at 440&nbsp;nm (aCDOM) and [secchi depth]. The measurement of these values will give an idea about the water quality of the water body being studied. A very high concentration of green pigments like chlorophyll might indicate the presence of an [algal bloom], for example, due to [eutrophication] processes. Thus, the chlorophyll concentration could be used as a proxy or indicator for the trophic condition of a water body. In the same manner, other optical quality parameters such as suspended particles or Suspended Particulate matter (SPM), Colored Dissolved Organic Matter (CDOM), Transparency (Kd), and chlorophyll-a (Chl-a) can be used to monitor water quality.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Overview

## SensorThings API  
### Overview  
SensorThings API is an Open Geospatial Consortium (OGC) standard providing an open and unified framework to interconnect IoT sensing devices, data, and applications over the Web. It is an open standard addressing the syntactic interoperability and semantic interoperability of the Internet of Things. It complements the existing IoT networking protocols such CoAP, MQTT, HTTP, 6LowPAN. While the above-mentioned IoT networking protocols are addressing the ability for different IoT systems to exchange information, OGC SensorThings API is addressing the ability for different IoT systems to use and understand the exchanged information. As an OGC standard, SensorThings API also allows easy integration into existing Spatial Data Infrastructures or Geographic Information Systems.
OGC SensorThings API has two parts: (1) Part I - Sensing and (2) Part II - Tasking. OGC SensorThings API Part I - Sensing was released for public comment on June 18, 2015. The OGC Technical Committee (TC) approves start of electronic vote on December 3, 2015, and the SensorThings API Part I - Sensing passed the TC vote on February 1, 2016. The official OGC standard specification was published online on July 26, 2016. In 2019 the SensorThings API was also published as a United Nation's ITU-T Technical Specification.
OGC SensorThings API Part II - Tasking Core was released for public comment on February 20, 2018, and it passed the TC vote on June 1, 2018. The official OGC standard specification for the SensorThings API Part II - Tasking Core was published online on January 8, 2019.
In order to offer a better developer experience, the SensorThings API Part II - Tasking Core Discussion Paper was published online on December 18, 2018. The Tasking Core Discussion paper provides 15 JSON examples showing how SensorThings API Part II - Tasking Core can be used.  
SensorThings API is an [Open Geospatial Consortium] (OGC) standard providing an open and unified framework to interconnect [Internet of Things|IoT] sensing devices, data, and applications over the Web. It is an [open standard] addressing the [Interoperability#Syntactic interoperability|syntactic interoperability] and [semantic interoperability] of the Internet of Things. It complements the existing IoT networking protocols such [CoAP], [MQTT], [Hypertext Transfer Protocol|HTTP], [6LoWPAN|6LowPAN]. While the above-mentioned IoT networking protocols are addressing the ability for different IoT systems to exchange information, OGC SensorThings API is addressing the ability for different IoT systems to use and understand the exchanged information. As an OGC standard, SensorThings API also allows easy integration into existing [Spatial Data Infrastructure]s or [Geographic Information Systems].  
OGC SensorThings API has two parts: (1) Part I - Sensing and (2) Part II - Tasking. OGC SensorThings API Part I - Sensing was released for public comment on June 18, 2015. The OGC Technical Committee (TC) approves start of electronic vote on December 3, 2015, and the SensorThings API Part I - Sensing passed the TC vote on February 1, 2016. The [http://docs.opengeospatial.org/is/15-078r6/15-078r6.html official OGC standard specification] was published online on July 26, 2016. In 2019 the SensorThings API was also published as a United Nation's ITU-T Technical Specification.  
OGC SensorThings API Part II - Tasking Core was released for public comment on February 20, 2018, and it passed the TC vote on June 1, 2018. The [http://docs.opengeospatial.org/is/17-079r1/17-079r1.html official OGC standard specification] for the SensorThings API Part II - Tasking Core was published online on January 8, 2019.  
In order to offer a better developer experience, [https://portal.opengeospatial.org/files/?artifact_id=79179 the SensorThings API Part II - Tasking Core Discussion Paper] was published online on December 18, 2018. The Tasking Core Discussion paper provides 15 JSON examples showing how SensorThings API Part II - Tasking Core can be used.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Design

### Design
SensorThings API is designed specifically for resource-constrained IoT devices and the Web developer community. It follows [REST] principles, the [JSON] encoding, and the OASIS [Open Data Protocol|OData] protocol and URL conventions. Also, it has an [MQTT] extension allowing users/devices to publish and subscribe updates from devices, and can use [CoAP] in addition to HTTP.
SensorThings API data model
The foundation of the SensorThings API is its data model that is based on the [ISO 19156] (ISO/OGC [Observations and Measurements]), that defines a conceptual model for observations, and for features involved in sampling when making observations. In the context of the SensorThings, the features are modelled as Things, Sensors (i.e., Procedures in O&M), and Feature of Interests. As a result, the SensorThings API provides an interoperable Observation-focus view, that is particularly useful to reconcile the differences between heterogeneous sensing systems (e.g., in-situ sensors and remote sensors).  
An IoT device or system is modelled as a Thing. A Thing has an arbitrary number of Locations (including 0 Locations) and an arbitrary number of Datastreams (including 0 Datastreams). Each Datastream observes one ObservedProperty with one Sensor and has many Observations collected by the Sensor. Each Observation observes one particular FeatureOfInterest. The O&M based model allows SensorThings to accommodate heterogeneous IoT devices and the data collected by the devices.  
SensorThings API provides two main functionalities, each handled by a part. The two profiles are the Sensing part and the Tasking part. The Sensing part provides a standard way to manage and retrieve observations and metadata from heterogeneous IoT sensor systems, and the Sensing part functions are similar to the OGC [Sensor Observation Service]. The Tasking part provides a standard way for parameterizing - also called tasking - of task-able IoT devices, such as sensors or actuators. The Tasking part functions are similar to the OGC [http://www.opengeospatial.org/standards/sps Sensor Planning Service].  The Sensing part is designed based on the ISO/OGC [Observations and Measurements] (O&M) model, and allows IoT devices and applications to CREATE, READ, UPDATE, and DELETE (i.e., HTTP POST, GET, PATCH, and DELETE) IoT data and metadata in a SensorThings service.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Entities (resources)

### Entities (resources)
SensorThings API Part I - Sensing defines the following resources. As SensorThings is a RESTful web service, each entity can be CREATE, READ, UPDATE, and DELETE with standard [HTTP Verbs|HTTP verbs] ([POST (HTTP)|POST], [HTTP GET|GET], PATCH, and DELETE):
* Thing: An object of the physical world (physical things) or the information world (virtual things) that is capable of being identified and integrated into communication networks.
* Locations: Locates the Thing or the Things it associated with.
* HistoricalLocations: Set provides the current (i.e., last known) and previous locations of the Thing with their time.
* Datastream: A collection of Observations and the Observations in a Datastream measure the same ObservedProperty and are produced by the same Sensor.
* ObservedProperty : Specifies the phenomenon of an Observation.
* Sensor : An instrument that observes a property or phenomenon with the goal of producing an estimate of the value of the property.
* Observation: Act of measuring or otherwise determining the value of a property.
* FeatureOfInterest: An Observation results in a value being assigned to a phenomenon.The phenomenon is a property of a feature, the latter being the FeatureOfInterest of the Observation.  
* TaskingCapabilities: Specifies the task-able parameters of an actuator.
* Tasks: A collection of Tasks that has been created.
* Actuator : A type of transducer that converts a signal to some real-world action or phenomenon.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Data array extensions

### Data array extensions
In order to reduce the data size transmitted over the network, SensorThings API data array extension allows users to request for multiple Observation entities and format the entities in the dataArray format. When a SensorThings service returns a dataArray response, the service groups Observation entities by Datastream or MultiDatastream, which means the Observation entities that link to the same Datastream or the same MultiDatastream are aggregated in one dataArray.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Evaluation

### Evaluation
Interoperability between OpenIoT and SensorThings
"We believe that the implementation of the SensorThing API will be a major improvement for the OpenIoT middleware. It will give OpenIoT a standardized and truly easy to use interface to sensor values.This will complement the rich semantic reasoning services with a simple resource based interface. And the consistent data model mapping gives both a common context to describe the internet of things".  
Efficiency of SensorThings API
A comprehensive evaluation of the SensorThings API is published in [http://www.mdpi.com/1424-8220/15/9/24343 Jazayeri, Mohammad Ali, Steve HL Liang, and Chih-Yuan Huang. "Implementation and Evaluation of Four Interoperable Open Standards for the Internet of Things." Sensors 15.9 (2015): 24343-24373].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Quotes

### Quotes
SensorThings API was demonstrated in a pilot project sponsored by the [Department of Homeland Security] [DHS Science and Technology Directorate|Science and Technology Directorate]. Dr. Reginald Brothers, the Undersecretary of the Homeland Security Science and Technology, was "impressed with the ‘state of the practical’ where these various industry sensors can be integrated today using open standards that remove the stovepipe limitations of one-off technologies. "

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | OGC SensorThings API standard specification

### OGC SensorThings API standard specification
* [http://docs.opengeospatial.org/is/15-078r6/15-078r6.html OGC® SensorThings API Part 1: Sensing] Whiskers is an OGC SensorThings API framework. It will have a [JavaScript] client and a light-weight server for IoT gateway devices (e.g., Raspberry Pi or BeagleBone). Whiskers aim to foster a healthy and open IoT ecosystem, as opposed to one dominated by proprietary information silos. Whiskers aims to make SensorThings development easy for the large and growing world of IoT developers.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | OGC SensorThings API standard specification | GOST

#### GOST
GOST is an open source implementation of the SensorThings API in the [Go (programming language)|Go programming language] initiated by Geodan. It contains easily deployable server software and a JavaScript client. Currently (June 2016) it is in development but a first version can already be downloaded and deployed. The software can be installed on any device supporting Docker or Go (e.g. Windows, Linux, Mac OS and Raspberry Pi). By default sensor data is stored in a [PostgreSQL] database.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | OGC SensorThings API standard specification | FROST

#### FROST
FROST-Server is an Open Source server implementation of the OGC SensorThings API. FROST-Server implements the entire specification, including all extensions. It is written in Java and can run in Tomcat or Wildfly and is available as a Docker image. Among its many features is the ability to use String or UUID based entity IDs.  
FROST-Client is a Java client library for communicating with a SensorThings API compatible server.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | OGC SensorThings API standard specification | SensorThings HcDT Charting SDK

#### SensorThings HcDT Charting SDK
SensorThings HcDT is a JavaScript charting library for the OGC SensorThings API. It is based on the open source Highcharts library and [https://datatables.net/ DataTables]. It is a front-end charting library enable developers to connect to datastreams from any OGC SensorThings API service, and display the sensor observations in charts, tables, or dashboard widgets for web applications.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | OGC SensorThings API standard specification | 52°North STA

#### 52°North STA
52N SensorThingsAPI is an open source implementation of the OGC SensorThings API. Its core features are the interoperability with the [https://github.com/52North/SOS/ 52N SOS] implementing the [Sensor Observation Service|OGC Sensor Observation Service], customizable database mappings and several convenience extensions. It can be deployed as a Docker container, inside an [Apache Tomcat] or as a standalone application.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | [https://www.dhs.gov/science-and-technology Department of Homeland Security S&T] Shaken Fury Operational Experiment

### Example applications  
#### [https://www.dhs.gov/science-and-technology Department of Homeland Security S&T] Shaken Fury Operational Experiment
In 2019 the Shaken Fury operational experiment for the DHS Next Generation First Responder program depicts a scenario of an earthquake causing partial structural collapse and HAZMAT leak at a stadium. OGC SensorThings API is used as the standard interface that interconnects multiple sensors and offers an IoT enabled real-time situational awareness.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | Smart Citizens for Smart Cities YYC - crowd-sourced air quality sensing

#### Smart Citizens for Smart Cities YYC - crowd-sourced air quality sensing
On Oct 8th 2016, a group of volunteers (smart citizens) in Calgary gathered together, assembled their own sensors, installed at their houses, and formed a crowd-sourced air quality sensor network. All data are publicly available via OGC SensorThings API. This citizen sensing efforts increased the number of Calgary's air quality sensors from 3 to more than 50.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | Smart Emission Project in Nijmegen, NL

#### Smart Emission Project in Nijmegen, NL
Smart emission is an air quality monitoring project in the city of Nijmegen, NL. The project deployed multiple air quality sensors throughout the city. Data are published with open standards, including OGC SensorThings API. Part of the project is an open source ETL engine to load the project sensor data into an OGC SensorThings API.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | SensorThings Dashboard

#### SensorThings Dashboard  
This [https://github.com/SensorThings-Dashboard/SensorThings-Dashboard dashboard] provides easy-to-use client-side visualisation of Internet-of-Things sensor data from OGC SensorThings API compatible servers. Various types of widgets can be arranged and configured on the dashboard. It is a [web application] and can be embedded into any website. A live demo is available on the [https://sensorthings-dashboard.github.io/ project page].
https://github.com/SensorThings-Dashboard/SensorThings-Dashboard

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | GOST Dashboard v2

#### GOST Dashboard v2
[https://github.com/gost/dashboard-v2 GOST Dashboard v2] is an open source library of custom HTML elements (web components) supporting SensorThings API. These elements facilitate the development of HTML applications integrating functionality and data from SensorThings API compatible services. The components are developed with [https://www.predix-ui.com Predix-UI] and [https://www.polymer-project.org/ Polymer].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Example applications | AFarCloud project OGC Connector

#### AFarCloud project OGC Connector
The connector enables interoperability between OGC-compliant data sources and the semantic middleware developed in the Horizon 2020 ECSEL [https://cordis.europa.eu/project/id/783221 project AFarCloud]. It is a modular Java application with Docker-based deployment, implemented according to the 15-078r6 OGC SensorThings API 1.0 Implementation Standard.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | SensorThings API | Comparison between OGC SensorThings API and OGC Sensor Observation Services

### Comparison between OGC SensorThings API and OGC Sensor Observation Services
SensorThings API provides functions similar to the OGC [Sensor Observation Service], an OGC specification approved in 2005. Both standard specifications are under the OGC [Sensor Web Enablement] standard suite. The following table summarizes the technical difference between the two specifications.  
{| class="wikitable"  
!   !! OGC SensorThings API !! OGC [Sensor Observation Service] (SOS)

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite Sentinel Project | Overview

## Satellite Sentinel Project  
### Overview  
The Satellite Sentinel Project (SSP) was conceived by George Clooney and Enough Project co-founder John Prendergast during their October 2010 visit to South Sudan. Through the use of satellite imagery, SSP provides an early warning system to deter mass atrocities in a given situation by focusing world attention  and generating rapid responses to human rights and human security concerns taking place in that situation.  
Evidence of Northern-aligned forces deployed to Abyei Region, Sudan<br/>(21 March 2011)
Satellite image of the burning of Tajalei, Sudan<br/>(6 March 2011)  
The Satellite Sentinel Project (SSP) was conceived by [George Clooney] and [Enough Project] co-founder [John Prendergast (activist)|John Prendergast] during their October 2010 visit to [South Sudan]. Through the use of satellite imagery, SSP provides an [early warning system] to deter mass atrocities in a given situation by focusing world attention  and generating rapid responses to human rights and human security concerns taking place in that situation.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite Sentinel Project | Activities

### Activities
SSP currently produces reports on the state of the conflict in the border regions between [Sudan] and [South Sudan]. [DigitalGlobe] provides [satellite imagery] and analysis. Their reporting is then released to the press and policymakers by the [Enough Project]. In 2011, the Satellite Sentinel Project detected images of freshly-dug mass grave sites in the [South Kordofan], a state of South Sudan, where [Sudan|Sudanese] military forces had killed members of a black ethnic minority suspected to support South Sudanese forces. SSP was the first to provide evidence consistent with the razing of the villages of Maker Abior, Todach, and Tajalei in the [Abyei] region of Sudan, and the project has discovered eight alleged [mass graves] in [South Kordofan], Sudan. SSP also planned to investigate how illegal trade in diamonds, gold and ivory was used to fund human-rights abusers.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite Sentinel Project | Organization and funding

### Organization and funding
[Not On Our Watch Project] provided [seed money] to launch the Satellite Sentinel Project.  The Enough Project contributes field reports, policy analysis and communications strategy, and, together with Not On Our Watch and its SUDANNOW partners, pressures policymakers by urging the public to act. [Google] and Internet strategy firm Trellon, LLC collaborate to design the web platform.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite Sentinel Project | Limits to effectiveness

### Limits to effectiveness
[Patrick Meier (humanitarian)|Patrick Meier], a [crisis mapping] expert, has observed that the deterrent value of any surveillance is diminished in the absence of consequences for the perpetrators of violence. Specific to Sudan, other technologies such as [Unmanned aerial vehicle|drones] are necessary to differentiate threats from nomads in order to generate actionable information.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Wide-area motion imagery | Overview

## Wide-area motion imagery  
### Overview  
Wide-area motion imagery (WAMI) is an approach to surveillance, reconnaissance, and intelligence-gathering that employs specialized software and a powerful camera system—usually airborne, and for extended periods of time—to detect and track hundreds of people and vehicles moving out in the open, over a city-sized area, kilometers in diameter. For this reason, WAMI is sometimes referred to as wide-area persistent surveillance (WAPS) or wide-area airborne surveillance (WAAS).
A WAMI sensor images the entirety of its coverage area in real time. It also records and archives that imagery in a database for real-time and forensic analysis. WAMI operators can use this live and recorded imagery to spot activity otherwise missed by standard video cameras with narrower fields of view, analyze these activities in context, distinguish threats from normal patterns of behavior, and perform the work of a larger force.
Military and security personnel are the typical users of WAMI, employing the technology for such missions as force protection, base security, route reconnaissance, border security, counter-terrorism, and event security. However, WAMI systems can also be used for disaster response, traffic pattern analysis, wildlife protection, and law enforcement.  
Wide-area motion imagery (WAMI) is an approach to [surveillance], [reconnaissance], and [Military intelligence|intelligence]-gathering that employs specialized software and a powerful camera system—usually airborne, and for extended periods of time—to [object detection|detect] and [video tracking|track] hundreds of people and vehicles moving out in the open, over a city-sized area, kilometers in diameter. For this reason, WAMI is sometimes referred to as wide-area persistent surveillance (WAPS) or wide-area airborne surveillance (WAAS).  
A WAMI sensor images the entirety of its coverage area in real time. It also records and archives that imagery in a database for real-time and forensic analysis. WAMI operators can use this live and recorded imagery to spot activity otherwise missed by standard video cameras with narrower fields of view, analyze these activities in context, distinguish threats from normal patterns of behavior, and perform the work of a larger force.  
Military and security personnel are the typical users of WAMI, employing the technology for such missions as force protection, base security, route reconnaissance, border security, counter-terrorism, and event security. However, WAMI systems can also be used for [disaster response], traffic pattern analysis, [wildlife protection], and law enforcement.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Wide-area motion imagery | Capabilities and enabling technologies

### Capabilities and enabling technologies
The typical WAMI sensor produces imagery at an update rate of 1&nbsp;Hz or faster from one or more multiple megapixel cameras. The system then seamlessly stitches together the collected images and applies algorithms to [Georeferencing|geo-register] them, ensuring that the sensor picture represents [ground truth].  
As far as resolution goes, WAMI systems usually have a 0.5 meter [Ground sample distance|ground sample distance (GSD)]—enough to detect and track moving targets throughout the scene. Should a user need to take a closer look at a subject, the WAMI system can cue other available sensors, such as hi-res full-motion video cameras, to make the identification.
Users can select different video streams pulled from the WAMI system's vast field of view and, with the help of advanced [data compression] techniques, watch them live on their computer screens or handheld devices. In some systems, users can also designate "watchboxes" within the sensor's field of view to provide automated alerts should the system detect movement in the area.  
All WAMI is tagged for time and location before being stored in an airborne or ground-based database. Users can remotely access this database and, similar to DVR functionality, can speed through or rewind the imagery to find specific incidents. In addition, just as with the real-time imagery, WAMI users can pan, tilt, and zoom within the archived imagery.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Wide-area motion imagery | Evolution of WAMI systems

### Evolution of WAMI systems
The very first WAMI system was developed in the early 2000s by a Lawrence Livermore National Laboratory team led by John Marion, as part of the Sonoma Persistent Surveillance Program. In 2005, the sensor transitioned to the [United States Department of Defense|U.S. Department of Defense], and in 2006, the Army sent the system—dubbed [Constant Hawk]—to Iraq on Short 360-300 [turboprop] aircraft as part of a Quick Reaction Capability. Three years later, Constant Hawk also deployed to Afghanistan.  
Weighing 1500 pounds, Constant Hawk initially comprised six electro-optical 11-megapixel cameras that covered 25 square kilometers. This payload was later upgraded to six 16-megapixel cameras.  
Since the deployment of Constant Hawk, WAMI systems have gotten smaller, lighter, and more capable. The current generation Kestrel Block II, for instance, employs eight electro-optical/infrared cameras that, together, form a 440-megapixel mosaic and cover 113 square kilometers. Yet this WAMI system weighs less than 85 pounds—light enough to be mounted on a tethered blimp, or [aerostat], which can be kept aloft for weeks at a time.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Wide-area motion imagery | List of WAMI systems

### List of WAMI systems
* [Constant Hawk]
* Angel Fire
* Blue Devil
* Lightweight Expeditionary Airborne Persistent Surveillance (LEAPS)
* Airborne Wide Area Persistent Surveillance Sensor (AWAPPS)
* [Kestrel (surveillance system)|Kestrel]
* Kestrel Block II (formerly KS-200)
* [ARGUS-IS|Autonomous Real-Time Ground Ubiquitous Surveillance Imaging System (ARGUS-IS)]
* [Gorgon Stare]
* [Redkite]
* Simera
* CorvusEye
* SkEye
* HawkEye II
* Heli-Tele
* WASP (IAI/TAMAM)
* POPSTAR (IAI/TAMAM)
* ESEN Systems Integration WAMI

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Paz (satellite) | Overview

## Paz (satellite)  
### Overview  
Paz (Spanish for "Peace") is a Spanish Earth observation and reconnaissance satellite launched on 22 February 2018. It is Spain's first spy satellite. The satellite is operated by Hisdesat. Paz was previously referred to as SEOSAR (Satélite Español de Observación SAR).  
Paz (Spanish for "Peace") is a Spanish Earth observation and reconnaissance satellite launched on 22 February 2018. It is Spain's first spy satellite. The satellite is operated by [Hisdesat]. Paz was previously referred to as SEOSAR (Satélite Español de Observación SAR).  
### Overview
For observational purposes, Paz uses a [synthetic aperture radar] (SAR) to collect images of Earth for governmental and commercial use, as well as other ship tracking and weather sensors, which enables high-resolution mapping of large geographical areas at day and night. The X-band radar imaging payload operates at a wavelength of , or a frequency of 9.65&nbsp;gigahertz.  
The Paz satellite is operated in a constellation with the German SAR fleet [TerraSAR-X] and [TanDEM-X] on the same orbit. The collaboration was agreed on by both Hisdesat and former European aerospace manufacturer [Astrium], operator of the two other satellites. The high-resolution images will be used for military operations, border control, intelligence, [environmental monitoring], protection of natural resources, city, and infrastructure planning, and monitoring of natural catastrophes.  
Originally, Paz was scheduled for launch from the [Yasny launch base], Russia, in 2014, but this was delayed due to Russia's 2014 annexation of Crimea, resulting in an [International Court of Arbitration] legal battle between [Hisdesat] and [Kosmotras]. The US launch was estimated to cost around , cost partially reduced by the inclusion of several mobile internet satellites on the same flight. The launch was shared by two  SpaceX test satellites for their [Starlink], named Tintin A and B. It was the final flight of a [Falcon 9 Block 3|Block 3] first stage, and reused the booster B1038 from the [Formosat-5] mission. and its total mass with fuel is . It also featured [Payload fairing|Fairing] 2.0 with a recovery attempt using a [crew boat] named Mr. Steven that is equipped with a net. The fairing narrowly missed the boat, leading to a soft water landing.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Paz (satellite) | Applications

### Applications
PAZ satellite images have been successfully used to monitor ground surface displacements, precipitation and cloud ice and crop classification.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Overview

## List of Earth observation satellites  
### Overview  
Earth observation satellites are Earth-orbiting spacecraft with sensors used to collect imagery and measurements of the surface of the earth. These satellites are used to monitor short-term weather, long-term climate change, natural disasters. Earth observations satellites provide information for research subjects that benefit from looking at Earth’s surface from above (such as meteorology, oceanography, terrestrial ecology, glaciology, atmospheric science, hydrology, geology, and many more). Types of sensors on these satellites include passive and active remote sensors. Sensors on Earth observation satellites often take measurements of emitted energy over some portion of the electromagnetic spectrum (e.g., UV, visible, infrared, microwave, or radio).
The invention of climate research through the use of satellite remote telemetry began in the 1960s through development of space probes to study other planets. During the U.S. economic decline in 1977, with much of NASA's money going toward the shuttle program, the Reagan Administration proposed to reduce spending on planetary exploration. During this time, new scientific evidence emerged from ice and sediment cores that Earth's climate had experienced rapid changes in temperature, running contrary to the previously held belief that the climate changed on a geological time scale.  These changes increased political interest in gathering remote-sensing data on the Earth itself and stimulated the science of climatology.  
True color image of the Earth from space. This image is a composite image collected over 16 days by the MODIS sensor on NASA’s [Terra (satellite)|Terra] satellite.
NASA Earth science satellite fleet as of September 2020, planned through 2023.
Earth observation satellite missions developed by the ESA as of 2019.  
[Earth observation satellites] are Earth-orbiting spacecraft with sensors used to collect imagery and measurements of the surface of the earth. These satellites are used to monitor short-term weather, long-term climate change, natural disasters. Earth observations satellites provide information for research subjects that benefit from looking at Earth’s surface from above (such as [meteorology], [oceanography], [terrestrial ecology], [glaciology], [atmospheric science], [hydrology], [geology], and many more). Types of sensors on these satellites include [Passive remote sensing|passive] and [Lidar|active] remote sensors. Sensors on Earth observation satellites often take measurements of emitted energy over some portion of the [electromagnetic spectrum] (e.g., UV, visible, infrared, microwave, or radio).  
The invention of [climate research] through the use of [satellite] remote [telemetry] began in the 1960s through development of space probes to study other planets. During the U.S. economic decline in 1977, with much of [NASA]'s money going toward the [Space Shuttle|shuttle] program, the Reagan Administration proposed to reduce spending on planetary exploration. During this time, new scientific evidence emerged from ice and sediment cores that Earth's climate had experienced rapid changes in temperature, running contrary to the previously held belief that the climate changed on a geological time scale.  These changes increased political interest in gathering remote-sensing data on the Earth itself and stimulated the science of [climatology].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Classification

### Classification
The lists below classify Earth observation satellites in two large groups: satellites operated by government agencies of one or more countries (public domain) versus commercial satellites built and maintained by companies (private domain). The satellite lifetime, between launch and reentry, is often called a satellite mission. These lists focus on currently active missions, rather than inactive retired missions or planned future missions. However, some examples of past and future satellites are included. Active, inactive, or planned classifications are relevant as of 2021.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Satellites launched by governmental agencies | Active government satellites

### Satellites launched by governmental agencies  
#### Active government satellites  
{|class="sortable wikitable sticky-header" style="font-size:90%;"
!Name
!Status
!Agency
!Launch
!class=unsortable|Description

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Satellites launched by governmental agencies | Inactive government satellites

#### Inactive government satellites  
{|class="sortable wikitable sticky-header" style="font-size:90%;"  
!Name
!Status
!Agency
!Launch
!Description

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Satellites launched by governmental agencies | Planned government satellites

#### Planned government satellites
{|class="sortable wikitable" style="font-size:90%;"
!Name
!Status
!Agency
!Description
!Launch date

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Private or commercial satellites | Active commercial satellites

### Private or commercial satellites  
#### Active commercial satellites
{|class="sortable wikitable" style="font-size:90%;"
!Name
!Status
!Owner/Agency
!Launch

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Private or commercial satellites | Inactive commercial satellites

#### Inactive commercial satellites
{|class="sortable wikitable" style="font-size:90%;"
!Name
!Status
!Owner/Agency
!Launch
!End of mission

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | Private or commercial satellites | Planned commercial satellites

#### Planned commercial satellites
{|class="sortable wikitable" style="font-size:90%;"
!Name
!Status
!Owner/Agency
!Launch
!End of mission
!Description

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | List of Earth observation satellites | See also

### See also
NASA Water and Energy Cycle satellite missions as of 2006.
NASA Earth science satellites as of 2017.  
*[Committee on Earth Observation Satellites]
*[Earth observation satellite]
*[First images of Earth from space]
*[Orbital spaceflight]
*[Satellite imagery#Imaging satellites|Imaging satellites]
*[Satellite]
*[Satellite imagery]
*[Timeline of Earth science satellites]
*[Uncrewed space mission]  
Related lists:
*[List of government space agencies]
*[List of orbits]
*[List of satellites in geosynchronous orbit]
*[List of uncrewed spacecraft by program]

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Overview

## Remote sensing in geology  
### Overview  
Remote sensing is used in the geological sciences as a data acquisition method complementary to field observation, because it allows mapping of geological characteristics of regions without physical contact with the areas being explored. About one-fourth of the Earth's total surface area is exposed land where information is ready to be extracted from detailed earth observation via remote sensing. Remote sensing is conducted via detection of electromagnetic radiation by sensors. The radiation can be naturally sourced (passive remote sensing), or produced by machines (active remote sensing) and reflected off of the Earth surface. The electromagnetic radiation acts as an information carrier for two main variables. First, the intensities of reflectance at different wavelengths are detected, and plotted on a spectral reflectance curve. This spectral fingerprint is governed by the physio-chemical properties of the surface of the target object and therefore helps mineral identification and hence geological mapping, for example by hyperspectral imaging. Second, the two-way travel time of radiation from and back to the sensor can calculate the distance in active remote sensing systems, for example, Interferometric synthetic-aperture radar. This helps geomorphological studies of ground motion, and thus can illuminate deformations associated with landslides, earthquakes, etc.
Remote sensing data can help studies involving geological mapping, geological hazards and economic geology (i.e., exploration for minerals, petroleum, etc.). These geological studies commonly employ a multitude of tools classified according to short to long wavelengths of the electromagnetic radiation which various instruments are sensitive to. Shorter wavelengths are generally useful for site characterization up to mineralogical scale, while longer wavelengths reveal larger scale surface information, e.g. regional thermal anomalies, surface roughness, etc. Such techniques are particularly beneficial for exploration of inaccessible areas, and planets other than Earth. Remote sensing of proxies for geology, such as soils and vegetation that preferentially grows above different types of rocks, can also help infer the underlying geological patterns. Remote sensing data is often visualized using Geographical Information System (GIS) tools. Such tools permit a range of quantitative analyses, such as using different wavelengths of collected data sets in various Red-Green-Blue configurations to produce false color imagery to reveal key features. Thus, image processing is an important step to decipher parameters from the collected image and to extract information.  
Richat Structure by [Shuttle Radar Topography Mission] (SRTM). Instead of being a [Impact event|meteorite impact], the landform is more likely to be a collapsed dome [Fold (geology)|fold structure]. |339x339px  
[Remote sensing] is used in the [Geology|geological sciences] as a data acquisition method complementary to [Fieldnotes|field observation], because it allows [Geological map|mapping] of geological characteristics of regions without physical contact with the areas being explored. About one-fourth of the Earth's total surface area is exposed land where information is ready to be extracted from detailed earth observation via remote sensing. Remote sensing is conducted via detection of [electromagnetic radiation] by sensors.  
Remote sensing data can help studies involving geological mapping, [Geologic hazards|geological hazards] and [economic geology] (i.e., exploration for minerals, petroleum, etc.). [Exploration geophysics|Geophysical methods], for instance [Sonar] and [Acoustics|acoustic methods], shares similar properties with remote sensing but electromagnetic wave is not the sole medium. Geotechnical instrumentations, for example [piezometer], [tiltmeter] and [Global Positioning System|Global Positioning System (GPS)], on the other hand, often refer to instruments installed to measure discrete point data, compared to imagery in remote sensing. The [thermal infrared] (TIR) region measures mainly emission while [microwave] region record [backscatter]ing portion of reflection. Data acquired from higher elevation captures a larger [field of view]/ spatial coverage, but the [Image resolution|resolutions] are often lower. Prior mission planning regarding flight path, weight load, carrying sensor etc. have to be done before deployment. Another parameter controlling the overall reflectance is [surface roughness].  
To identify mineral, available spectral reflectance libraries, for instance the [https://speclab.cr.usgs.gov/spectral-lib.html USGS Spectral Library], summarize diagnostic absorption bands for many materials not limited to rocks and minerals. This helps create a mineral map to identify the type of mineral sharing similar spectra, with minimal in situ field work. The different approaches are summarized and classified in literature but unfortunately there is no universal recipe for mineral identification.  
For rocks, be they [Igneous rock|igneous], [Sedimentary rock|sedimentary] or [Metamorphic rock|metamorphic], most of their diagnostic spectral characteristics of mineralogy are present in longer wavelength (SWIR and TIR), which is for [http://www.ga.gov.au/webtemp/image_cache/GA7833.pdf example] present in the [Advanced Spaceborne Thermal Emission and Reflection Radiometer|ASTER] mission.
*Carbonate Index (CI): D13/D14
*Quartz Index (QI): D11*D11 / D10*D12
*Mafic Index (MI): D12/D13
Hyperspectral imaging gives high spectral resolution, but as a trade-off the spatial and radiometric resolutions are lower

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Overview | Soil

##### Soil
Surficial soil is a good proxy to the geology underneath. Some of the properties of soil, alongside lithology mentioned above, are retrievable in remote sensing data, for instance Landsat ETM+, to develop the [soil horizon] and therefore aid its [Soil classification|classification]. for long-term change detection. Another approach is surface energy balance model, which makes prediction on the actual [evapotranspiration]. This see-through characteristic, notably of the L-band (1.25&nbsp;GHz) microwave of 1–2&nbsp;m penetration, allows subsurface mapping and possibly identification of past aquifer. The area, however, could possibly serve as one spectacular example of intracontinental volcanism or [Hotspot (geology)|hotspot].  
Water bodies, for instance ocean and glaciers, could be studied using remote sensing. Here are two examples for plankton and glacier mapping.  
The bloom of photosynthesizing phytoplankton is an ecological proxy to favorable environmental conditions. Satellite remote sensing in VNIR wavelength region help locate sporadic event of change in ocean colour due to relative increase in related absorption in spectral curve. Some notable applications include mapping of clean-ice and debris-covered glaciers, glacier fluctuation records, mass balance and volume change studies to aid generating topographic map and quantitative analysis.
thumb|353x353px|U.S. Seismic Hazard Maps 2014

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Overview | Earthquakes

#### Earthquakes
[Earthquake]s manifest itself in movement of earth surface. Remote sensing can also help earthquake studies by two aspects. One is to better understand the local ground condition. For instance some soil type, which is prone to [liquefaction] (e.g. saturated loose alluvial material), do more damage under vibration and therefore earthquake hazard zoning may help in reducing property loss. SAR interferometry is a technology utilizing multiple SAR images recording the backscattering of microwave signal. Aside from these long-lived thermal fields, there are some positive thermal anomalies of 3–4&nbsp;°C on land surface or around −5&nbsp;°C for sea water in earthquake [epicenter] areas. The contrast appears 7–14 days prior to the earth movement. Though the observation is supported by laboratory experiments, the possible causes of these differences are still debatable. Remote sensing of mangrove and vegetation as a natural barrier to manage risks therefore becomes a hot topic. The recent advancement and development is highly anticipated in the near future, especially as hyperspectral imaging system and very high resolution (up to sub meter grade) satellite images prevails. New classification schemes distinguishing species from composition could be developed for environmental studies. Split-based approach to divide large images into subimages for further analysis by redefining change detection threshold have reduced computation time and have shown to be consistent with manual mapping of affected areas. [Seismicity] is considered geophysical method on the other hand. The data could be collected throughout the eruption cycle, from unrest, to eruption then relaxation. and CO2 are also possible candidates for volcanic monitoring these days. For example, the use of aerial photos to update landslide inventory is popular in [Hong Kong] landslide studies. The [LiDAR] technique to create a [Digital elevation model|High Resolution Digital Elevation Model (HRDEM)] and [Digital terrain model|Digital Terrain Model (DTM)] with vegetation cover is crucial to the quantification of [slope], [Aspect (geography)|slope aspect], [stream power], [drainage density] and many more parameters for landslide hazard models. Furthermore, InSAR maps enable the semi-automatic identification, mapping, and classification of landslides. The hazard risk management could be further discussed using [Geographic information system|geographical information system (GIS)].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Overview | Land subsidence

#### Land subsidence
Land [subsidence] primarily consists of the lowering of the ground surface. [Interferometric synthetic-aperture radar] enables accurate, precise, and cost-effective measurement and monitoring of this phenomenon over wide areas. Aerial [Lidar|LiDAR] has also been successfully used for monitoring large-scale land subsidence in wide areas.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Economic geology

### Economic geology
In the context of [economic geology], the surficial data help locate possible reserves of [natural resource]s. One point to bear in mind is the inherit limitation, that remote sensing is for surface detection while natural resources are concentrated in depth, therefore its use is somewhat limited. Nonetheless, there are some proxies providing valuable inputs, including the following examples

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Economic geology | Groundwater investigations

#### Groundwater investigations
Targeting [groundwater] resources for supply is one of the ultimate goals in water management. While much of the information is indeed provided from [hydrogeology], geophysical methods and drilling, the remote sensing technique, using the same principle to integrate data collected for the surface, can infer possible confined/ unconfined [aquifers]. For instance in radar data ([ground penetrating radar]), which is able to penetrate the ground deep into meters, may show some diffuse reflection for ‘rough’ surface relative to the wavelength used. The change in lithology may suggest soft rock and unconsolidated sediments, where [porosity] is high.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Vegetation

### Vegetation
The surficial vegetation studies are mainly accomplished by multispectral or hyperspectral image analysis, mainly because of the lower penetration depth and higher resolutions of VNIR wavelength region. new observations point out that the temporal evolution of spectral ratio between 2:5 micrometer thermal emissions (thermal signature) could infer eruption modes, from lava fountain down to silicic lava flows. Recent suggestion have been made to improve the spatial resolution to locate more accurately the heat source vent, so as to elucidate the unsolved puzzle of the volcanology, which is strongly related to the [tidal heating] caused by the [orbital eccentricity] of [Jupiter]. Modeling has shown that a suitable distance between the surveyed ground and the sensor has to be maintained to ensure a meaningful pixel size to resolve the Io surface. Remote sensing by satellite also reduces jittering as the sensor is held stable in space and gives accurate data in the absence of atmosphere for terrestrial observations, notwithstanding the strong [radiation zone] in Jupiter which dramatically limits sensor lifetime. All these promotes future instrumentation and orbit design.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Image processing

### Image processing
> Main: Digital image processing  
[Digital image processing|Image processing] is crucial to convert raw data into useful information. For imaging remote sensing, where spectral data are collected and recorded in [pixel]s of an [image], a two dimensional representation. After removal of noise and [calibration], images are then [Georeferencing|geo-referenced] to relate [pixel] to real-life geography. The first-hand data are then corrected to remove noise such as atmospheric disturbance, structural effects and [distortion]. Remote sensing data are often validated by [ground truth], which usually serves as [Training, test, and validation sets|training data] in image classification to ensure quality. Remote sensing collected data for geology and lineament density while GIS derived drainage density, topography elevation, gradient, landuse and the annual rainfall data. Besides the examples in Europe, landslides in Hong Kong brought casualties and property damage to the territory before the establishment of relevant government organization to carry out systematic studies to reduce risk of slope failure. The major contributing factors, similar to landslides all over the world, include geology, discontinuities (structural), weathering and [rainfall]. The intense rainfall (>2000mm/year) rapidly raises the [pore pressure] due to [Infiltration (hydrology)|infiltration]. While local hydrogeological models generated with the aid of in situ, for instance, [Piezometer|piezometric measurements] and discontinuity mapping, could help elucidate the kinematics of landslides, employing remote sensing for landslide evaluation in Hong Kong is never short of experience. For instance, [Interferometric synthetic-aperture radar] and aerial photo interpretation is the tool used in history for detecting surface deformation and updating landslide inventory respectively. GIS is also used to overlay layers of terrain (elevation and slope angle), lithology with rainfall data to generate landslide hazard maps. With the different weightings in respective parameters, the hazard could be zoned to get the risk controlled.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in geology | Image processing | Urban environmental applications

#### Urban environmental applications
Remote sensing has much potential in environmental applications. To name a few, the land use planning (for instance nuclear power plant location & dumping sites), monitoring of soil erosion and atmospheric pollution, vegetation etc. have been in great interest in the recent decade.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Spectroradiometry for Earth and planetary remote sensing | Overview

## Spectroradiometry for Earth and planetary remote sensing  
### Overview  
Spectroradiometry is a technique in Earth and planetary remote sensing, which makes use of light behaviour, specifically how light energy is reflected, emitted, and scattered by substances, to explore their properties in the electromagnetic (light) spectrum and identify or differentiate between them. The interaction between light radiation and the surface of a given material determines the manner in which the radiation reflects back to a detector, i.e., a spectroradiometer. Combining the elements of spectroscopy and radiometry, spectroradiometry carries out precise measurements of electromagnetic radiation and associated parameters within different wavelength ranges. This technique forms the basis of multi- and hyperspectral imaging and reflectance spectroscopy, commonly applied across numerous geoscience disciplines, which evaluates the spectral properties exhibited by various materials found on Earth and planetary bodies.
Spectral properties such as brightness and reflectance patterns vary depending on the mineralogical compositions and crystalline structures of the given material. This variation is contributed by the presence of spectrally active components within the material, such as metallic oxides and clay minerals, which give rise to unique absorption features. Upon measurements with a spectroradiometer, these absorption features can be quantified as characteristic absorption bands in a reflectance spectra. The specific shapes associated with the bands that occur at distinctive wavelength positions enable the identification of minerals and facilitate lithological interpretations.
Conventionally, spectroradiometry is applied to the following portions of wavelengths in the electromagnetic (light) spectrum:  
Ultraviolet (UV): 1 nm – 400 nm
Visible-near Infrared (VNIR): 400 nm – 750 nm
Short-wave Infrared (SWIR): 750 nm – 2500 nm
Mid Infrared (MIR): 2500 nm – 5000 nm
Thermal Infrared (TIR): 7500 nm – 15000 nm
Today, most geological applications with spectroradiometry are focused within the visible-near infrared and short-wave infrared wavelength ranges. Spectroradiometry offers a simple, non-destructive, rapid, and efficient approach that complements traditional and heavy-duty geochemical methods, to characterize mineral assemblages and rock textures. It thereby facilitates the study of geological processes, exploration for natural resources, and reconstruction of past environments and climates. Its application extends not only to Earth but also to extraterrestrial planets, broadening our understanding of geological processes beyond our own planet.  
{| class="wikitable infobox" style="width: auto; border:1px solid #ADD8E6;" cellspacing="0" cellpadding="0"  
[Portal:Earth sciences|Earth Sciences Portal]  
[Geology]  
[Remote sensing (geology)|Remote Sensing]  
:  [Spectroradiometer]  
:  [Spectroscopy]  
:  [Radiometry]  
Spectroradiometry is a technique in Earth and planetary remote sensing, which makes use of [light] behaviour, specifically how [light|light energy] is [reflected], emitted, and [scattering|scattered] by substances, to explore their properties in the [electromagnetic spectrum|electromagnetic (light) spectrum] and identify or differentiate between them. The interaction between [radiation|light radiation] and the surface of a given material determines the manner in which the radiation reflects back to a detector, i.e., a [spectroradiometer]. Combining the elements of [spectroscopy] and [radiometry], spectroradiometry carries out precise measurements of [electromagnetic radiation] and associated parameters within different [wavelength] ranges. This technique forms the basis of [multispectral imaging|multi-] and [hyperspectral imaging] and [reflectance|reflectance spectroscopy], commonly applied across numerous [geology|geoscience] disciplines, which evaluates the spectral properties exhibited by various materials found on Earth and planetary bodies.  
Spectral properties such as [brightness] and [reflectance] patterns vary depending on the [mineralogy|mineralogical compositions] and [Crystallography|crystalline structures] of the given material. Spectroradiometry offers a simple, non-destructive, rapid, and efficient approach that complements traditional and heavy-duty [geochemistry|geochemical methods], to characterize [mineral|mineral assemblages] and [Texture (geology)|rock textures]. It thereby facilitates the study of geological processes, exploration for [natural resources], and reconstruction of [natural environment|past environments] and [paleoclimatology|climates].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Spectroradiometry for Earth and planetary remote sensing | How spectroradiometry works

### How spectroradiometry works
In spectroradiometry, spectral features can be recognized and quantified by making use of the [Spectrum|spectra] containing different parameters measured by [spectroradiometer]s. Values of spectral parameters like [reflectance] can then be directly extracted from all [pixels] in the [digital image|imagery], aggregated and averaged to produce a [reflectance] curve for spectral analysis.  In both scenarios, the [spectroradiometer]s are frequently [calibration|calibrated] with a white [diffusion] [reflectance] panel, which provides a reference [reflectance] value (99%) to maintain experimental accuracy.  
In order to facilitate data analysis, the raw [reflectance] spectra are commonly [normalization (image processing)|normalized] to provide better visualization and quantification of trends and patterns of spectral parameters. This is commonly done by [statistical] techniques including detrending and [Continuum (measurement)|continuum] removal.
A comparison of a [[reflectance spectra for a given material before and after the continuum removal process, modified from Tan et al., 2021. A similar example of such indices is the [Normalized difference vegetation index] (NDVI).  
For example, [water]-bearing minerals commonly share distinctive [Absorption (electromagnetic radiation)|absorption] features indicating the presence of [hydroxyl groups] (-OH) and [molecular] [water] (H2O), which include the asymmetrical [Absorption (electromagnetic radiation)|absorption] features due to [overtones] near 1400&nbsp;nm (AS1400), as well as [absorption bands|absorption peaks] near 1900 (D1900) and 2200&nbsp;nm (D2200). With higher [molecular] [water] contents, the AS1400 feature becomes more asymmetrical, the [Absorption (electromagnetic radiation)|absorption] near 2200&nbsp;nm strengthens, but the one near 1900&nbsp;nm weakens.  
In practice, however, certain minerals may exhibit [Absorption (electromagnetic radiation)|absorption features] that coincide with those of water in similar [em spectrum|wavelength] intervals.  
Considering the identification capabilities of spectroradiometry for different [minerals] and [rocks], the comprehensive databases that encompass spectral signatures are crucial. Such databases serve as valuable resources which contributes to advancing our understanding and characterization of [Earth materials]. The USGS spectral library provides a collection of reflectance spectra for rock-forming minerals and other Earth materials, spanning from [ultraviolet] (350&nbsp;nm) to [infrared|shortwave infrared] (SWIR) regions (2500&nbsp;nm). Likewise, the ECOSTRESS spectral library integrates spectral data from multiple spectral libraries, consolidating information on minerals and rocks into a standardized data format.  
(1) [Clay minerals] ([Phyllosilicates]) The [volcanic ash|ashes] typically have a high purity, composed of [silicates] (such as [quartz]), and [phyllosilicates] (such as [kaolinite], [Serpentine subgroup|serpentine]), which are highly sensitive towards spectral parameters such that they demonstrate characteristic spectral features when compared to the background [sediments]. [Volcanic ash]es with a high [silica] content, known as [felsic] ashes, stand out from the background sediments due to their high [albedo] and [reflectance] values. Meanwhile, [nonimaging optics|non-imaging] spectroradiometry, combined with [field research|field] scanning and sampling, is suitable for localized applications, providing [geochronology|age] implications and constraints for [stratigraphic] units. Importantly, its applicability to inaccessible areas further expands its utility in assessing and investigating Earth's valuable [natural resource|resources]. Higher concentrations of [illite] may indicate areas conducive to [ore] precipitation, and the spectral characteristics of [illite], including strong [absorption bands|absorption] features near 1400, 1900, and 2200&nbsp;nm (D1400, D1900, D2200) in the [em spectrum|wavelength spectra], can be utilized to identify and trace [ore] [fluid] pathways and [Deposition (geology)|deposition]. The intense [weathering] processes occurring in [granitic] rocks give rise to the [denudation] and [Leaching (chemistry)|leaching] of major element [oxides], leaving behind the highly decomposed [regolith]. One specific [Rare-earth element|REE] of interest is [neodymium] (Nd), which has extensive applications in the industry. Making use of these spectral characteristics, combined with [geochemical] interpretations and [machine learning], the identification and [mapping (cartography)|mapping] of Nd-enriched [regolith] areas can be fostered, which may provide implications towards potential [Rare-earth element|REE] mineralization and respective [ore] bodies. Some of the signature spectral features of [hydrocarbon] molecules are as follows: Spectroradiometry, with its ability to characterize surficial compositions and study the [geology] of these [celestial bodies], is considered a key technique in [planetary science].  
In recent years, huge efforts are devoted to the exploration of [Mars], especially on its [geology of Mars|geology], which helps unravelling the planet's evolutionary history, understanding past and ongoing events occurring on the planet, and providing insights towards its [habitability] for human exploration. Each of these minerals are found in different regions on [Mars], and are detected by [spectroradiometer]s through their characteristic absorption features on the [reflectance] spectra.  
* [Olivine]: Broad absorption features centred near 1000&nbsp;nm. The features get deeper and wider with increasing [iron] contents.
* [Pyroxene]: Broad absorption features near 1000 and 2000&nbsp;nm. Most [pyroxenes] in Mars are [calcium] depleted ([dunite], [pyroxenite]) which slightly shifts the absorption features towards shorter wavelengths.
* [Plagioclase]: Broad absorption features centred near 1300&nbsp;nm given that there is [Substitution reaction|substitution] of [iron] and [calcium] ions.
The [mafic] [silicates] made up the composition of [basaltic] [crust (geology)|crusts] on Mars. At Martian [valleys] and [craters], such minerals are often seen associated with [hydrated] [silica] deposits resulted from [metasomatism|alteration].
visible and near infrared (VNIR) [reflectance] spectra of common [mafic] [silicates] on Mars, modified from Viviano et al., 2014.
* [Monochromator]: Captures and splits incoming [radiation] (polychromatic light) into ranges of different [wavelengths] (monochromatic light) to foster spectral analysis. In contrast, [non-imaging optics|non-imaging] spectroradiometers capture the spectral properties of the entire field of view without spatial variations. Many [non-imaging optics|non-imaging] spectroradiometers are relatively smaller in sizes and utilized in [land|ground-based] applications. Some are used in [laboratories] while some are portable and can be used in the [field research|field]. In general, 4 kinds of resolutions are commonly specified for each spectroradiometer.  
A visualization of spectral resolution. The area bounded by the curve represents the magnitude of [electromagnetic radiation] reflected by a given material at various [wavelength]s. Devices with high [spectral resolution] can measure the [reflectance] for the material within narrow bands of [wavelength].|313x313px
[Spectral resolution|Spectral resolution] concerns the capability of a [sensor] in a [spectroradiometer] to measure the [light|light intensity] according to specific [wavelengths] on the [electromagnetic spectrum]. It is related to the amount of spectral detail to be detected in each [spectral band] so as to discriminate among different materials. Described by the amount, [wavelength] interval, and [Bandwidth (signal processing)|width] of spectral bands in which the sensor conducts wavelength measurements, a sensor with high spectral resolution would mean that it is able to capture a spectrum of light and divides it into hundreds or thousands of narrow [spectral bands] or channels with typical [Bandwidth (signal processing)|widths] up to 10 and 20&nbsp;nm.  
multi- and [hyperspectral imaging]. A hyperspectral sensor collects spectral data in a continuous spectrum whereas a multispectral sensor collects spectral data in varying [Bandwidth (signal processing)|bandwidths] in the [EM spectrum].|268x268px]]
In modern times, [multispectral imaging|multi-] and [hyperspectral imaging] [sensors] are mainly adopted in spectroradiometry. Unlike ordinary [broadband] sensors which possess only a few [spectral bands] for measurements, they enable the extraction of spectral properties in sufficiently high [spectral resolution]s, allowing for the detection and analysis of diagnostic [absorption (electromagnetic radiation)|absorption] features in a continuous spectrum. [hyperspectral imaging|Hyperspectral] sensors divide the detected light intensity into many, narrow, and contiguous (i.e., adjacent) [spectral bands] to reconstruct a full [spectrum], while [multispectral imaging|multispectral] [sensors] measures [light|light intensity] using [spectral bands] of varying [Bandwidth (signal processing)|bandwidths] in the [electromagnetic spectrum|wavelength spectrum] which might not be contiguous. Consequently, a hyperspectral sensor is often regarded as having greater [spectral resolution] in comparison to a multispectral sensor, hence a better potential in mineralogical diagnosis and [lithology] [mapping (cartography)|mapping].  
A visualization of spatial resolution, which refers to the level of detail or the smallest discernible features that can be captured by a given [spectroradiometer].|268x268px
[Spatial resolution|Spatial resolution] evaluates the quality of an image captured by [imaging] spectroradiometers. It describes the extent of spatial detail the sensors can record, i.e., the smallest feature detected, based on [pixel] and [grid (graphic design)|grid] sizes of the captured [digital image]ry.  A sensor with fine spatial resolution would capture an image with small grid cells, thus recording more spatial details and image pixels.  
A visualization of radiometric resolution. The area bounded by the curve represents the magnitude of [electromagnetic radiation] reflected by a given material at various [wavelength]s. Devices with high [radiometric resolution] can precisely measure and detect relatively small differences in the values of [reflectance] for a given material.|295x295px
[Radiometric resolution|Radiometric resolution] deals with the sensitivity of a sensor towards measuring the magnitude of [electromagnetic radiation] and [light|light intensity]. A sensor with high radiometric resolution can detect and discriminate subtle variations in [brightness] and [radiation] magnitudes.  In the context of [multispectral imaging], the greater the number of data bits per [pixel|pixel (bit depth)] of the [image] recorded, the better the quality and interpretability of the [image], thus the finer the radiometric resolution.  
[Temporal resolution|Temporal resolution] is the [frequency] or the repeat cycle of a sensor, most commonly referring to [sensors] on [imaging] spectroradiometers, to capture [images] and acquire spectral information. An [imaging] spectroradiometer with high temporal resolution typically requires less time to complete spectral measurements of an [image].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Spectroradiometry for Earth and planetary remote sensing | How spectroradiometry works | Spectroradiometers in practice

#### Spectroradiometers in practice
The following table shows the categories and some examples of [spectroradiometer]s worldwide which are commonly used for spectral data collection in geoscience studies.  
{| class="wikitable"  
!Spectroradiometer
!Category
!Resolution
!Primary applications  
* Spaceborne
* [Multispectral imaging]  
* [Spectral resolution]: 250 m (bands 1–2); 500 m (bands 3–7); and 1000 m (bands 8–36)
* [Temporal resolution]: 1-2 Days  
* Study of Earth's [geomorphology|surface processes]
* Monitoring of [volcanism|volcanic activity]  
* Spaceborne (onboard [Terra satellite])
* [Multispectral imaging]  
* [Spectral resolution]: 15 m (bands 1–3); 30 m (bands 4–9); and 90 m (bands 10–14)
* [Temporal resolution]: 4-16 Days  
* [geomorphology|Surface] [lithology] mapping and [mineral] detection
* Combined with [digital elevation model] for 3D analysis of land cover  
* Spaceborne
* [Multispectral imaging]  
* [Spatial resolution]: 1.1&nbsp;km  
* Study of Earth's [geomorphology|surface processes]
* Monitoring of environmental changes  
* Airborne
* [Hyperspectral imaging]  
* [Spectral resolution]: 15&nbsp;nm
* [Spatial resolution]: 20 m  
* [Multispectral pattern recognition] for land cover changes
* [lithology] and [mineral] mapping  
* Ground-based
* [Hyperspectral imaging]  
* [Spectral resolution]: 2.8&nbsp;nm (700&nbsp;nm wavelength intervals); 8&nbsp;nm (1500&nbsp;nm wavelength intervals); 6&nbsp;nm (2100&nbsp;nm wavelength intervals)  
* High resolution [lithology] and [mineral] mapping
* [Mining] and [ore] exploration
* [Soil] analysis

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Spectroradiometry for Earth and planetary remote sensing | Other applications of spectroradiometry

### Other applications of spectroradiometry
* Ground truthing
* [Soil] analysis and monitoring
* Forest canopy and [vegetation] studies
* Landscape ecology studies
* [Agricultural] studies
* [Biodiversity] conservation
* [Water quality] assessments
* [Camouflage] characterization and detection

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Spectroradiometry for Earth and planetary remote sensing | See also

### See also
* [Spectroradiometer]
* [Radiometry]
* [Multispectral imaging]
* [Hyperspectral imaging]
* [Reflectance]
* [Ultraviolet-visible spectroscopy]
* [Infrared spectroscopy]

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Oversampled binary image sensor | Overview

## Oversampled binary image sensor  
### Overview  
An oversampled binary image sensor is an image sensor with non-linear response capabilities reminiscent of traditional photographic film. Each pixel in the sensor has a binary response, giving only a one-bit quantized measurement of the local light intensity. The response function of the image sensor is non-linear and similar to a logarithmic function, which makes the sensor suitable for high dynamic range imaging.  
An oversampled binary image sensor is an [image sensor] with non-linear response capabilities reminiscent of traditional [photographic film]. Each pixel in the sensor has a binary response, giving only a one-bit quantized measurement of the local light intensity. The response function of the image sensor is non-linear and similar to a logarithmic function, which makes the sensor suitable for [high dynamic range imaging]. During exposure, each micron-sized grain has a binary fate: Either it is struck by some incident photons and becomes "exposed", or it is missed by the photon bombardment and remains "unexposed". In the subsequent film development process, exposed grains, due to their altered chemical properties, are converted to silver metal, contributing to opaque spots on the film; unexposed grains are washed away in a chemical bath, leaving behind the transparent regions on the film. Thus, in essence, photographic film is a binary imaging medium, using local densities of opaque silver grains to encode the original light intensity information. Thanks to the small size and large number of these grains, one hardly notices this quantized nature of film when viewing it at a distance, observing only a continuous gray tone.  
The oversampled binary image sensor is reminiscent of photographic film. Each pixel in the sensor has a binary response, giving only a one-bit quantized measurement of the local light intensity. At the start of the exposure period, all pixels are set to 0. A pixel is then set to 1 if the number of photons reaching it during the exposure is at least equal to a given threshold q. One way to build such binary sensors is to modify standard memory chip technology, where each memory [bit cell] is designed to be sensitive to visible light. With current CMOS technology, the level of integration of such systems can exceed 109~1010 (i.e., 1 giga to 10 giga) pixels per chip. In this case, the corresponding pixel sizes (around 50~nm ) are far below the diffraction limit of light, and thus the image sensor is [oversampling] the optical resolution of the light field. Intuitively, one can exploit this spatial redundancy to compensate for the information loss due to one-bit quantizations, as is classic in oversampling [delta-sigma converter]s.  
Building a binary sensor that emulates the photographic film process was first envisioned by [Eric Fossum|Fossum], who coined the name digital film sensor (now referred to as a quanta image sensor). The original motivation was mainly out of technical necessity. The [miniaturization] of camera systems calls for the continuous shrinking of pixel sizes. At a certain point, however, the limited [full-well capacity] (i.e., the maximum photon-electrons a pixel can hold) of small pixels becomes a bottleneck, yielding very low [signal-to-noise ratio]s (SNRs) and poor [dynamic range]s. In contrast, a binary sensor whose pixels need to detect only a few photon-electrons around a small threshold q has much less requirement for full-well capacities, allowing pixel sizes to shrink further.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Oversampled binary image sensor | Imaging model | Lens

### Imaging model  
#### Lens
Fig.1 The imaging model. The simplified architecture of a diffraction-limited imaging system. Incident light field  \lambda_0(x)  passes through an optical lens, which acts like a linear system with a diffraction-limited point spread function (PSF). The result is a smoothed light field  \lambda(x) , which is subsequently captured by the image sensor.
Consider a simplified camera model shown in Fig.1. The  \lambda_0(x)  is the incoming light intensity field. By assuming that light intensities remain constant within a short exposure period, the field can be modeled as only a function of the spatial variable  x . After passing through the optical system, the original light field  \lambda_0(x)  gets filtered by the lens, which acts like a linear system with a given [impulse response]. Due to imperfections (e.g., aberrations) in the lens, the impulse response, a.k.a. the [point spread function] (PSF) of the optical system, cannot be a Dirac delta, thus, imposing a limit on the resolution of the observable light field. However, a more fundamental physical limit is due to light [diffraction]. As a result, even if the lens is ideal,  the PSF is still unavoidably a small blurry spot. In optics, such diffraction-limited spot is often called the [Airy disk], sensor, with a spatial resolution of 32×32 pixels. The final image (lower-right corner) is obtained by incorporating 4096 consecutive frames, 11 of which are shown in the figure.]]  
One of the most important challenges with the use of an oversampled binary image sensor is the reconstruction of the light intensity  \lambda(x)  from the binary measurement  b_m . [Maximum likelihood|Maximum likelihood estimation] can be used for solving this problem.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Overview

## Satellite imagery  
### Overview  
Satellite images (also Earth observation imagery, spaceborne photography, or simply satellite photo) are images of Earth collected by imaging satellites operated by governments and businesses around the world. Satellite imaging companies sell images by licensing them to governments and businesses such as Apple Maps and Google Maps.  
The first images from space were taken on the sub-orbital V-2 rocket flight launched by the [US] on October 24, 1946.
Satellite image of Fortaleza
Satellite images (also Earth observation imagery, spaceborne photography, or simply satellite photo) are [image]s of [Earth] collected by [imaging satellite]s operated by governments and businesses around the world. Satellite imaging companies sell images by licensing them to governments and businesses such as [Apple Maps] and [Google Maps].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | History

### History
> Further: First images of Earth from space  
The first crude image taken by the satellite Explorer 6 shows a sunlit area of the central [Pacific Ocean] and its cloud cover. The photo was taken when the satellite was about  above the surface of the Earth on August 14, 1959. At the time, the satellite was crossing [Mexico].
The first images from space were taken on [Sub-orbital spaceflight|sub-orbital flights]. The US-launched [V-2] flight on October 24, 1946, took one image every 1.5 seconds. With an [Apsis|apogee] of 65 miles (105&nbsp;km), these photos were from five times higher than the previous record, the 13.7 miles (22&nbsp;km) by the Explorer II balloon mission in 1935. The first satellite (orbital) photographs of Earth were made on August 14, 1959, by the U.S. [Explorer 6]. The first satellite photographs of the [Moon] might have been made on October 6, 1959, by the [Soviet] satellite [Luna 3], on a mission to photograph the far side of the Moon. [The Blue Marble] photograph was taken from space in 1972, and has become very popular in the media and among the public. Also in 1972 the United States started the [Landsat program], the largest program for acquisition of imagery of Earth from space. In 1977, the first real time satellite imagery was acquired by the United States' [KH-11] satellite system. The most recent Landsat satellite, [Landsat 9], was launched on 27 September 2021.  
The first television image of Earth from space transmitted by the TIROS-1 weather satellite in 1960
All satellite images produced by [NASA] are published by [NASA Earth Observatory] and are freely available to the public. Several other countries have satellite imaging programs, and a collaborative European effort launched the [European Remote-Sensing Satellite|ERS] and [Envisat] satellites carrying various sensors. There are also private companies that provide commercial satellite imagery. In the early 21st century satellite imagery became widely available when affordable, easy to use software with access to satellite imagery databases was offered by several companies and organizations.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Applications

### Applications
Satellite images have numerous applications in a variety of fields.
* [Weather]: They guide meteorologists in forecasting patterns, tracking storms, and understanding [climate change].
* [Oceanography]: By measuring sea temperatures and monitoring ecosystems, satellite images unlock insights into our oceans' health and global climate.
* [Agriculture] and [fishing]: Satellite data helps locate fish populations, assess crop health, and optimize resource use for a thriving agricultural and fishing industry.
* [Biodiversity]: Conservation efforts leverage satellite technology to map habitats, monitor ecosystem changes, and protect endangered species.
* [Forestry]: Satellite data empowers sustainable forestry by tracking deforestation, assessing fire risks, and managing resources effectively.
* [Landscape]: Analyzing land use patterns with satellite images supports urban planning and facilitates sustainable development initiatives.
Less mainstream uses include anomaly hunting, a criticized investigation technique involving the search of satellite images for unexplained phenomena.  
The [Electromagnetic spectrum|spectrum] of satellite images includes visible light, near-infrared light, infrared light and radar, and many others. This wide range of light [frequencies] can provide researchers with large volumes of information. In addition to the satellite applications mentioned above, this data can serve as educational tools, advance scientific research and grow a deeper understanding of our environment.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Usage

### Usage
Satellite images are used in many fields of activity — agriculture, geological and [hydrological] research, forestry, environmental protection, territorial planning, educational, intelligence and military purposes. Such images can be made in the visible part of the spectrum, as well as in the ultraviolet, infrared and other parts of the range. There are also various terrain maps made using radar surveys.  
Currently, the decryption and analysis of satellite images is increasingly performed using automated software systems such as ERDAS Imagine or ENVI. At the beginning of the development of this industry, some of the types of image enhancements commissioned by the US government were performed by contractor firms. For example, ESL Incorporated has developed one of the first two-dimensional Fourier transforms for digital image processing.  
Satellite image analysis is actively used to protect the environment, for example, the "Visual satellite search for illegal landfills" method has identified more than 200 unauthorized municipal solid and household waste landfills on the territory of 5 subjects of the Russian Federation,. In France, satellite imagery has been used to spot private swimming pools, limiting evasion from a tax on swimming pools.  
The purchase of private imagery is also a common practice among the open-source intelligence (or [OSINT]) community. For example, it enables the estimation of the remaining soviet military hardware in storage that can be refitted in the context of the War in Ukraine.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Disadvantages

### Disadvantages
Composite image of Earth at night, as only half of Earth is at night at any given moment
Because the total area of the land on Earth is so large and because resolution is relatively high, satellite databases are huge and [image processing] (creating useful images from the raw data) is time-consuming. Preprocessing, such as [image destriping], is often required. Depending on the [sensor] used, weather conditions can affect image quality. For example, it is difficult to obtain images for areas of frequent cloud cover such as mountaintops. For such reasons, publicly available satellite image datasets are typically processed for visual or scientific commercial use by third parties.  
Commercial satellite companies do not place their imagery into the [public domain] and do not sell their imagery; instead, one must acquire a [license] to use their imagery. Thus, the ability to legally make derivative works from commercial satellite imagery is diminished.  
[Privacy] concerns have been brought up by some who wish not to have their property shown from above. Google Maps responds to such concerns in their [FAQ] with the following statement: "We understand your privacy concerns... The images that Google Maps displays are no different from what can be seen by anyone who flies over or drives by a specific geographic location."

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Data characteristics

### Data characteristics  
There are five types of [Image resolution|resolution] when discussing satellite imagery in remote sensing:  spatial, spectral, temporal, radiometric and geometric.  Campbell (2002) defines these as follows:
* Spatial resolution is defined as the [pixel] size of an image representing the size of the [surface area] (i.e. m2) being measured on the ground, determined by the sensors' [instantaneous field of view] (IFOV).
* Spectral resolution is defined by the [wavelength] interval size (i.e. the size of discrete segments of the electromagnetic spectrum) and the number of intervals that the sensor is measuring.
* Temporal resolution is defined by the amount of time (e.g. days) that passes between imagery collection periods for a given surface location.
*Radiometric resolution is defined as the ability of an imaging system to record many levels of brightness (e.g. [Contrast (vision)|contrast]) and to the effective bit-depth of the sensor (number of grayscale levels) and is typically expressed as 8-bit (0–255), 11-bit (0–2047), 12-bit (0–4095) or 16-bit (0–65,535).
* Geometric resolution refers to the satellite sensor's ability to effectively image a portion of the Earth's surface in a single pixel and is typically expressed in terms of [ground sample distance] (GSD). GSD is a term containing the overall optical and systemic noise sources and is useful for comparing how well one sensor can "see" an object on the ground within a single pixel. For example, the GSD of Landsat is ≈30m, which means the smallest unit that maps to a single pixel within an image is ≈30m x 30m. The latest commercial satellite (GeoEye 1) has a GSD of 0.41 m. This compares to a 0.3 m resolution obtained by some early military film based [reconnaissance satellite]s such as [Corona (satellite)|Corona].  
The resolution of satellite images varies depending on the instrument used and the altitude of the satellite's orbit. For example, the [Landsat] archive offers repeated imagery at 30 meter resolution for the planet, but most of it has not been processed from the raw data. [Landsat 7] has an average return period of 16 days. For many smaller areas, images with resolution as fine as 41&nbsp;cm can be available.  
Satellite imagery is sometimes supplemented with [aerial photography], which has higher resolution, but is more expensive per square meter. Satellite imagery can be combined with vector or raster data in a [Geographic information system|GIS] provided that the imagery has been spatially rectified so that it will properly align with other data sets.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain

#### Public domain
Satellite imaging of the Earth surface is of sufficient public utility that many countries maintain satellite imaging programs. The United States has led the way in making these data freely available for scientific use. Some of the more popular programs are listed below, recently followed by the [European Union]'s Sentinel constellation.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | CORONA

##### CORONA
The [Corona (satellite)|CORONA] program was a series of American strategic [reconnaissance satellite]s produced and operated by the [Central Intelligence Agency] (CIA) [Directorate of Science & Technology] with substantial assistance from the [United States Air Force|U.S. Air Force]. The type of imagery is [wet film] [panoramic] and it used two cameras (AFT&FWD) for capturing [stereographic] imagery.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | Landsat

##### Landsat
[Landsat program|Landsat] is the oldest continuous Earth-observing satellite imaging program. Optical Landsat imagery has been collected at 30 m resolution since the early 1980s. Beginning with [Landsat 5], thermal infrared imagery was also collected (at coarser spatial resolution than the optical data). The [Landsat 7], [Landsat 8], and [Landsat 9] satellites are currently in orbit.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | MODIS

##### MODIS
[Moderate Resolution Imaging Spectroradiometer|MODIS] has collected near-daily satellite imagery of the earth in 36 spectral bands since 2000. MODIS is on board the NASA Terra and Aqua satellites.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | Sentinel

##### Sentinel
The ESA is currently developing the [Sentinel (satellite)|Sentinel] constellation of satellites. Currently, 7 missions are planned, each for a different application. [Sentinel-1] (SAR imaging), [Sentinel-2] (decameter optical imaging for land surfaces), and [Sentinel-3] (hectometer optical and thermal imaging for land and water) have already been launched.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | ASTER

##### ASTER
The [Advanced Spaceborne Thermal Emission and Reflection Radiometer|ASTER] is an imaging instrument onboard Terra, the flagship satellite of NASA's Earth Observing System (EOS) launched in December 1999. ASTER is a cooperative effort between NASA, [Ministry of Economy, Trade and Industry|Japan's Ministry of Economy, Trade and Industry] (METI), and [Japan Space Systems] (J-spacesystems). ASTER data is used to create detailed maps of land surface temperature, reflectance, and elevation. The coordinated system of EOS satellites, including Terra, is a major component of NASA's Science Mission Directorate and the Earth Science Division. The goal of NASA Earth Science is to develop a scientific understanding of the Earth as an integrated system, its response to change, and to better predict variability and trends in climate, weather, and natural hazards.  
* Land surface climatology—investigation of land surface parameters, [Satellite temperature measurements|surface temperature], etc., to understand land-surface interaction and energy and moisture fluxes
* Vegetation and ecosystem dynamics—investigations of vegetation and soil distribution and their changes to estimate biological productivity, understand land-atmosphere interactions, and detect ecosystem change
* [Volcano monitoring]—monitoring of eruptions and precursor events, such as gas emissions, eruption plumes, development of lava lakes, eruptive history and eruptive potential
* Hazard monitoring—observation of the extent and effects of wildfires, flooding, [coastal erosion], earthquake damage, and tsunami damage
* [Hydrology]—understanding global energy and hydrologic processes and their relationship to global change; included is evapotranspiration from plants
* [Geology] and soils—the detailed composition and geomorphologic mapping of surface soils and bedrocks to study land surface processes and Earth's history
* Land surface and land cover change—monitoring [desertification], deforestation, and urbanization; providing data for conservation managers to monitor protected areas, national parks, and wilderness areas

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | Meteosat

##### Meteosat
Model of a first generation Meteosat geostationary satellite
The [Meteosat]-2 geostationary weather satellite began operationally to supply imagery data on 16 August 1981. [Eumetsat] has operated the Meteosats since 1987.  
*The [Meteosat visible and infrared imager] (MVIRI), three-channel imager: visible, infrared and water vapour; It operates on the first generation Meteosat, Meteosat-7 being still active.
*The 12-channel Spinning Enhanced Visible and Infrared Imager (SEVIRI) includes similar channels to those used by MVIRI, providing continuity in climate data over three decades; [Meteosat Second Generation] (MSG).
*The Flexible Combined Imager (FCI) on [Meteosat#Third Generation ("MTG")|Meteosat Third Generation] (MTG) will also include similar channels, meaning that all three generations will have provided over 60 years of climate data.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Public domain | Himawari

##### Himawari
The [Himawari (satellites)|Himawari] satellite series features advanced imaging technology and frequent data updates. Himawari-8 and Himawari-9 have become indispensable tools for weather forecasting, disaster management, and climate research, benefiting not only Japan but the entire Asia-Pacific region.  
*Frequent Updates: These satellites can provide full-disk images of the Asia-Pacific region every 10 minutes, and even more frequently (every 2.5 minutes) for specific areas (Japan), ensuring that meteorologists have up-to-date information for accurate weather forecasting.
*[Spectral band|Spectral Bands]:
**Visible Light Bands (0.47 μm, 0.51 μm, 0.64 μm): These bands are used for daytime cloud, land, and ocean surface observations. They provide high-resolution images that are critical for tracking cloud movements and assessing weather conditions.
**Near-Infrared Bands (0.86 μm, 1.6 μm, 2.3 μm, 6.9 μm, 7.3 μm, 8.6 μm, 9.6 μm, 11.2 μm, 13.3 μm): These bands help in distinguishing between different types of clouds, vegetation, and surface features. They are particularly useful for detecting fog, ice, and snow.
**Infrared Bands (3.9 μm, 6.2 μm, 10.4 μm, 12.4 μm): The remaining bands cover the thermal infrared spectrum. These bands are crucial for measuring cloud-top temperatures, sea surface temperatures, and atmospheric water vapor content. They enable continuous monitoring of weather patterns.
*Advanced Imaging Technology: Himawari-8 and Himawari-9 are equipped with the [https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html Advanced Himawari Imager (AHI)], which provides high-resolution images of the Earth. The AHI can capture images in 16 different spectral bands, allowing for detailed observation of weather patterns, clouds, and environmental phenomena.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Private domain | GeoEye

##### GeoEye  
GeoEye's [GeoEye-1] satellite was launched on September 6, 2008. The GeoEye-1 satellite has high resolution imaging system and is able to collect images with a ground resolution of 0.41&nbsp;meters (16&nbsp;inches) in [panchromatic] or black and white mode. It collects multispectral or color imagery at 1.65-meter resolution or about 64&nbsp;inches.
WorldView-2 image of Weston-super-Mare

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Private domain | Maxar

##### Maxar  
Maxar's [WorldView-2] satellite provides high resolution commercial satellite imagery with 0.46 m spatial resolution (panchromatic only). The 0.46 meters resolution of WorldView-2's panchromatic images allows the satellite to distinguish between objects on the ground that are at least 46&nbsp;cm apart. Similarly Maxar's [QuickBird] satellite provides 0.6 meter resolution (at [nadir]) panchromatic images.  
Maxar's [WorldView-3] satellite provides high resolution commercial satellite imagery with 0.31 m spatial resolution. WVIII also carries a short wave infrared sensor and an atmospheric sensor.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Private domain | Airbus Intelligence

##### Airbus Intelligence
Pleiades image of Central Park in [New York City]  
[Pléiades (satellite)|Pléiades] [Satellite constellation|constellation] is composed of two very-high-resolution (50 centimeters pan & 2.1 meter spectral) optical [Earth-imaging satellite]s. Pléiades-HR 1A and Pléiades-HR 1B provide the coverage of Earth's surface with a repeat cycle of 26 days. Designed as a dual civil/military system, Pléiades will meet the space imagery requirements of [Europe]an defense as well as civil and commercial needs.
is the advanced optical constellation, with four identical 30-cm resolution satellites with fast reactivity.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Private domain | Spot Image

##### Spot Image
SPOT image of Bratislava
Satellite view of Southern Luzon taken by the [ISS]
The 3 [SPOT (satellite)|SPOT satellites] in orbit (Spot 5, 6, 7) provide very high resolution images – 1.5 m for Panchromatic channel, 6m for Multi-spectral (R,G,B,NIR). Spot Image also distributes multiresolution data from other optical satellites, in particular from Formosat-2 ([Taiwan]) and Kompsat-2 ([South Korea]) and from radar satellites (TerraSar-X, ERS, Envisat, Radarsat). [Spot Image] is also the exclusive distributor of data from the high resolution [Pleiades satellites] with a resolution of 0.50 meter or about 20&nbsp;inches. The launches occurred in 2011 and 2012, respectively. The company also offers infrastructures for receiving and processing, as well as added value options.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | Imaging satellites | Private domain | Planet Labs

##### Planet Labs
[Planet Labs] operates three satellite imagery constellations, [RapidEye], [Flock-1|Dove] and [SkySat].  
In 2015, Planet acquired [BlackBridge], and its constellation of five RapidEye satellites, launched in August 2008. The RapidEye constellation contains identical [multispectral] sensors which are equally calibrated. Therefore, an image from one satellite will be equivalent to an image from any of the other four, allowing for a large amount of imagery to be collected (4 million km2 per day), and daily revisit to an area. Each travel on the same orbital plane at 630&nbsp;km, and deliver images in 5 meter pixel size. RapidEye satellite imagery is especially suited for agricultural, environmental, cartographic and disaster management applications. The company not only offers their imagery, but consults their customers to create services and solutions based on analysis of this imagery. The RapidEye constellation was retired by Planet in April 2020.  
Planet's Dove satellites are [CubeSat]s that weigh ,  in length, width and height, orbit at a height of about  and provide imagery with a resolution of  and are used for environmental, humanitarian, and business applications.  
SkySat image of [São Paulo], the largest city in Brazil
SkySat is a constellation of sub-metre resolution [Earth observation satellite]s that provide imagery, high-definition video and analytics services. Planet acquired the satellites with their purchase of Terra Bella (formerly Skybox Imaging), a [Mountain View, California]-based company founded in 2009 by Dan Berkenstock, Julian Mann, John Fenwick, and Ching-Yu Hu, from Google in 2017.  
The SkySat satellites are based on using inexpensive automotive grade electronics and fast commercially available processors, but scaled up to approximately the size of a [Refrigerator|minifridge]. The satellites are approximately  long, compared to approximately  for a 3U CubeSat, and weigh .

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Satellite imagery | See also

### See also
* [Aerial photography]
* [Earth observation satellite]
* [Moderate-resolution imaging spectroradiometer]
* [Reconnaissance satellite]
* [Remote sensing]
* [Shuttle Radar Topography Mission]
* [Timeline of first images of Earth from space]
* [Virtual globe]
** [NASA World Wind]
* [Weather satellite]

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Overview

## Remote sensing in archaeology  
### Overview  
Remote sensing techniques in archaeology are an increasingly important component of the technical and methodological tool set available in archaeological research. The use of remote sensing techniques allows archaeologists to uncover unique data that is unobtainable using traditional archaeological excavation techniques.  
Remote sensing techniques in archaeology are an increasingly important component of the technical and methodological tool set available in [archaeology|archaeological] research. The use of [remote sensing] techniques allows archaeologists to uncover unique data that is unobtainable using traditional archaeological [Excavation (archaeology)|excavation] techniques.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | General techniques

### General techniques
Remote Sensing methods employed in the service of archaeological investigations include:
*[Aerial archaeology|Aerial], UAV and Satellite Imaging
**[Aerial photography]
***[Multispectral Scanner|Multispectral] and [Hyperspectral imaging|Hyperspectral Sensors]
***[Multispectral Scanner|Thermal Infrared Multispectral Scanner (TIMS)]
***[Infrared photography|Color Infrared Film (CIR)]
***[Radar|Microwave Radar]
**Satellite Imaging
***[LIDAR|Laser altimeters or light detection and ranging (LIDAR)]
***[Synthetic aperture radar|Synthetic Aperture Radar (SAR)]
***[Interferometric synthetic aperture radar|INSAR - Interferometric SAR]
Ground-based geophysical methods such as [Ground-penetrating radar|Ground Penetrating Radar] and [Magnetometer|Magnetometry] are also used for archaeological imaging. Although these are sometimes classed as remote sensing, they are usually considered a separate discipline (see [Geophysical survey (archaeology)]).

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Satellite archaeology

### Satellite archaeology
Satellite archaeology is an emerging field of archaeology that uses high resolution satellites with thermal and infrared capabilities to pinpoint potential sites of interest in the earth around a meter or so in depth. The infrared light used by these satellites have longer wavelengths than that of visible light and are therefore capable of penetrating the Earth's surface. The images are then taken and processed by an archaeologist who specializes in satellite remote sensing in order to find any subtle anomalies on the Earth's surface.  
Landscape features such as soil, vegetation, geology, and man-made structures of possible cultural interest have specific signatures that the multi-spectral satellites can help to identify. The satellites can then make a 3D image of the area to show if there are any man-made structures beneath soil and vegetation that can not be seen by the naked eye. Commercially available satellites have a .4m-90m resolution that make it possible to see most ancient sites and their associated features in such places as Egypt, Perù and Mexico. It is a hope of archaeologists that in the next few decades resolutions will improve to the point where they are capable of zooming in on a single pottery shard buried beneath the earth's surface.  
Satellite archaeology is a non-invasive method for mapping and monitoring potential archaeological sites in an ever changing world that faces issues such as [urbanization], [looting], and [groundwater pollution] that could pose threats to such sites. In spite of this, satellites in archaeology are mostly a tool for broad scale survey and focused excavation. All archaeological projects need ground work in order to verify any potential findings.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Examples of regional applications | Maya research

### Examples of regional applications  
#### Maya research  
Some of the most prominent remote sensing research has been done in regard to [Maya civilization|Maya] studies in [Mesoamerica]. The [Petén Basin|Petén] region of northern [Guatemala] is of particular focus because [remote sensing] technology is of very definite use there. The Petén is a densely forested region and it lacks modern settlements and infrastructure. As a result, it is extremely difficult to survey, and because of this remote sensing offers a solution to this research problem. The use of remote sensing techniques in this region is a great example of the applications these methods have for archaeologists. The Petén is a hilly, [karst]ic, thickly forested landscape which offers an incredible barrier for field archaeologists to penetrate. With the advent of remote sensing techniques, a plethora of information has been uncovered about the region and about the people that inhabited it.  
The Petén is arguably one of the most difficult of the Maya landscapes in which to subsist. It is questions regarding [List of subsistence techniques|subsistence patterns] and related problems that have driven remote sensing methodology in the hopes of understanding the complex adaptations that the Maya developed. Remote sensing methods have also proven invaluable when working to discover [Feature (archaeology)|features], [cisterns], and [temples]. Archaeologists have identified vegetative differentiation associated with such features. With the advent of remote sensing, archaeologists are able to pinpoint and study the features hidden beneath this canopy without ever visiting the jungle.  
A pioneer in the use of remote sensing in Maya research is [NASA] archaeologist Tom Sever, who has applied remote sensing to research in Maya site discovery as well as mapping [causeways] ([sacbe]ob) and roads. Sever has stressed the enormous use of remote sensing in uncovering settlement patterns, population densities, societal structure, communication, and transportation. Sever has done much of his research in the Petén region of northern Guatemala, where he and his research team have used satellite imagery and [GIS] to map undiscovered roads and causeways the ancient [Maya civilization|Maya] built to connect cities and settlements. These landscape artifacts represent the advantage of using [remote sensing] as these [causeways] are not visible from the ground. By mapping these forms, Sever is able to locate new sites and further uncover ancient Maya methods of communicated and transportation. Sever and his team also use remote sensing methods to gather data on [deforestation]. The [rain forests] of the Petén are undergoing massive deforestation, and Sever's remote sensing offers another window into this understanding and halting this problem. Monitoring the rate of deforestation not only has important ecological value, but the use of remote sensing can detect landscape change. By measuring the magnitude of landscape change in terms of vegetative cover and [soil geography], as well as shifting land use patterns and the associated cultural diversity, archaeologists are given a window into depletion rates and trends in anthropogenic landscape alteration.  
Much attention has been devoted to the mapping of [canals] and [irrigation] systems. Synthetic Aperture Radar (SAR) has proved particularly useful in this research. SAR is a type of radar that is sensitive to linear and geometric features on the ground.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Examples of regional applications | Maya research | Maya "collapse"

##### Maya "collapse"  
One of Sever's research goals is understanding the comparatively sudden decline of many Maya centers in the central Lowlands region by the end of the 1st millennium [Common Era|CE], a happenstance often referred to as the "[Classic Maya collapse|(Classic) Maya collapse]". Sever's research on communication and transportation systems points to an extensive societal infrastructure capable of supporting the building and maintenance of the [causeways] and roadways. Using [satellite imagery], researchers have been able to map [canals] and [reservoirs]. These offer a glimpse into Maya cultural adaptations during the period of their highest population density. At the height of the classic period, the population in the Maya lowlands was 500 - 1300 people per square mile in rural areas, and even more in [urban region]s. This far outweighs the carrying capacity for this region, but this follows centuries of successful adaptation. Other data shows that by the end of the classic period, the Maya had already depleted much of the rain forest.  Understanding how the ancient Maya adapted to this [karst] topography could shed light on solutions to modern ecological problems that modern peoples in the Petén currently face, which is much the same, except there are fewer people who are causing even more damage to the biodiversity and cultural diversity. Sever believes that the Maya collapse was a primarily ecological disaster. By detecting deforestation rates and trends can help us to understand how these same processes affected the Maya.
An important contribution to the study of Maya has been provided by LiDAR thanks to its ability to penetrate dense tropical canopies. LiDAR has been applied to the site of Caracol, Belize in 2009, revealing an impressive monumental complex covered by jungle.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Examples of regional applications | Satellite archaeology in Peru

#### Satellite archaeology in Peru
In Peru, an  Italian scientific mission of CNR, directed by [Nicola Masini], provided important results by using satellite imagery for both site discovery and  the protection of archaeological heritage. In particular, by processing QuickBird images a large buried  settlement, including a pyramid, in the Nasca riverbed (Southern Peru), near the Ceremonial Center of [Cahuachi], has been detected. In the region of Lambayeque (Northern Peru), which is strongly affected by clandestine excavations, satellite imagery have been also employed for mapping and monitoring archaeological looting.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Examples of regional applications | Location of ancient Iram

#### Location of ancient Iram  
[Iram of the Pillars] is a [lost city] (or region surrounding the lost city) on the [Arabian Peninsula]. In the early 1980s a group of researchers interested in the history of Iram used [NASA] remote sensing satellites, [ground penetrating radar], [Landsat program] data and images taken from the [Space Shuttle Challenger|Space Shuttle Challenger] as well as [SPOT (satellites)|SPOT] data to identify old [camel train] routes and points where they converged. These roads were used as [frankincense] trade routes around 2800 BC to 100 BC.  
One area in the [Dhofar] province of [Oman] was identified as a possible location for an outpost of the lost civilization. A team including adventurer [Ranulph Fiennes], archaeologist [Juris Zarins], filmmaker [Nicholas Clapp], and lawyer [George Hedges], scouted the area on several trips, and stopped at a water well called Ash Shisar. Near this oasis was located a site previously identified as the 16th century Shis'r fort. Excavations uncovered an older settlement, and artifacts traded from far and wide were found.  This older fort was found to have been built on top of a large limestone cavern which would have served as the water source for the fort, making it an important oasis on the trade route to Iram.  As the residents of the fort consumed the water from underground, the water table fell, leaving the limestone roof and walls of the cavern dry. Without the support of the water, the cavern would have been in danger of collapse, and it seems to have done so some time between 300-500 AD, destroying the oasis and covering over the water source.  
Four subsequent excavations were conducted by Dr. [Juris Zarins], tracing the historical presence by the people of [ʿĀd], the assumed ancestral builders of Iram.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Remote sensing in archaeology | Examples of regional applications | Egypt and the Roman Empire

#### Egypt and the Roman Empire  
Archaeologist Dr [Sarah Parcak] uses satellites to search for sub-surface remains, as described in her [TED (conference)|TED Talk] on the [https://www.ted.com/talks/sarah_parcak_archaeology_from_space subject of space archaeology] and uses of [citizen science]. Parcak uses these satellites to hunt to for lost settlements, tombs, and pyramids in [Egypt]'s [Nile Delta]. She has also prospectively identified several significant sites in various parts of the ancient [Roman Empire].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Mohammed VI (satellites) | Overview

## Mohammed VI (satellites)  
### Overview  
The Mohammed VI satellites are a series of two Moroccan Earth observation and reconnaissance satellites, namely Mohammed VI-A and Mohammed VI-B, developed and built by Airbus Defence and Space and Thales Alenia Space based upon the Astrosat-1000 satellite bus. They are Morocco's first optical imaging satellites, and are operated by Morocco's Ministry of Defense, with an expected service life of 5 years. They are named after Mohammed VI, the King of Morocco.  
The Mohammed VI satellites are a series of two [Morocco|Moroccan] [Earth observation satellite|Earth observation] and [reconnaissance] satellites, namely Mohammed VI-A and Mohammed VI-B, developed and built by [Airbus Defence and Space] and [Thales Alenia Space] based upon the [Astrosat-1000] satellite bus. They are Morocco's first optical imaging satellites, and are operated by Morocco's Ministry of Defense, with an expected service life of 5 years. They are named after [Mohammed VI of Morocco|Mohammed VI], the King of Morocco.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Mohammed VI (satellites) | Design | Propulsion

#### Propulsion
The satellites have four [hydrazine] thrusters for [Reboost|reboosting] its orbit and keeping its altitude.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Mohammed VI (satellites) | Launch | Mohammed VI-A

### Launch  
#### Mohammed VI-A  
The Mohammed VI-A satellite, Morocco's first spy satellite, was launched on Vega flight VV11 on board the Vega launcher from Guiana Space Centre ELA-1, [French Guiana], on November 8, 2017. It was launched to low Earth [Sun-synchronous orbit] with an inclination of 97.9°.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Mohammed VI (satellites) | Launch | Mohammed VI-B

#### Mohammed VI-B  
The Mohammed VI-B satellite, Morocco's second spy satellite, was launched a year later on Vega flight VV14 from Guiana Space Centre ELA-1, French Guiana, on November 28, 2018. It was launched to Low Earth Sun-synchronous orbit with an inclination of 97.9°.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Image scaling | Applications | Pixel-art scaling

#### Pixel-art scaling
> Main: Pixel-art scaling algorithms  
As [pixel art|pixel-art] graphics are usually low-resolution, they rely on careful placement of individual pixels, often with a limited palette of colors. This results in graphics that rely on stylized visual cues to define complex shapes with little resolution, down to individual pixels. This makes scaling pixel art a particularly difficult problem.  
Specialized algorithms were developed to handle pixel-art graphics, as the traditional scaling algorithms do not take perceptual cues into account.  
Since a typical application is to improve the appearance of [history of video game consoles (fourth generation)|fourth-generation] and earlier [video game]s on [arcade emulator|arcade] and [console emulator]s, many are designed to run in real time for small input images at 60 frames per second.  
On fast hardware, these algorithms are suitable for gaming and other real-time image processing. These algorithms provide sharp, crisp graphics, while minimizing blur. Scaling art algorithms have been implemented in a wide range of emulators such as HqMAME and [DOSBox], as well as 2D [game engine]s and [game engine recreation]s such as [ScummVM]. They gained recognition with gamers, for whom these technologies encouraged a revival of 1980s and 1990s gaming experiences.  
Such filters are currently used in commercial emulators on [Xbox Live], [Virtual Console], and [PlayStation Network|PSN] to allow classic low-resolution games to be more visually appealing on modern [High-definition video|HD] displays. Recently released games that incorporate these filters include [Sonic's Ultimate Genesis Collection], [Castlevania: The Dracula X Chronicles], [Castlevania: Symphony of the Night], and [Akumajō Dracula X Chi no Rondo].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Overview

## Hyperspectral imaging  
### Overview  
Hyperspectral imaging collects and processes information from across the electromagnetic spectrum. The goal of hyperspectral imaging is to obtain the spectrum for each pixel in the image of a scene, with the purpose of finding objects, identifying materials, or detecting processes. There are three general types of spectral imagers. There are push broom scanners and the related whisk broom scanners (spatial scanning), which read images over time, band sequential scanners (spectral scanning), which acquire images of an area at different wavelengths, and snapshot hyperspectral imagers, which uses a staring array to generate an image in an instant.
Whereas the human eye sees color of visible light in mostly three bands (long wavelengths, perceived as red; medium wavelengths, perceived as green; and short wavelengths, perceived as blue), spectral imaging divides the spectrum into many more bands. This technique of dividing images into bands can be extended beyond the visible. In hyperspectral imaging, the recorded spectra have fine wavelength resolution and cover a wide range of wavelengths. Hyperspectral imaging measures continuous spectral bands, as opposed to multiband imaging which measures spaced spectral bands.
Engineers build hyperspectral sensors and processing systems for applications in astronomy, agriculture, molecular biology, biomedical imaging, geosciences, physics, and surveillance. Hyperspectral sensors look at objects using a vast portion of the electromagnetic spectrum. Certain objects leave unique "fingerprints" in the electromagnetic spectrum. Known as spectral signatures, these "fingerprints" enable identification of the materials that make up a scanned object. For example, a spectral signature for oil helps geologists find new oil fields.  
Two-dimensional projection of a hyperspectral cube.  
Hyperspectral imaging collects and processes information from across the [electromagnetic spectrum]. The goal of hyperspectral imaging is to obtain the spectrum for each pixel in the image of a scene, with the purpose of finding objects, identifying materials, or detecting processes. There are three general types of spectral imagers. There are [push broom scanner]s and the related [whisk broom scanner]s (spatial scanning), which read images over time, band sequential scanners (spectral scanning), which acquire images of an area at different wavelengths, and [snapshot hyperspectral imaging|snapshot hyperspectral imagers], which uses a [staring array] to generate an image in an instant.  
Whereas the [human eye] sees color of [visible light] in mostly [trichromatism|three bands] (long wavelengths, perceived as red; medium wavelengths, perceived as green; and short wavelengths, perceived as blue), spectral imaging divides the spectrum into many more bands. This technique of dividing images into bands can be extended beyond the visible. In hyperspectral imaging, the recorded spectra have fine wavelength resolution and cover a wide range of wavelengths. Hyperspectral imaging measures continuous spectral bands, as opposed to [multispectral image|multiband imaging] which measures spaced spectral bands.  
Engineers build hyperspectral sensors and processing systems for applications in astronomy, agriculture, molecular biology, biomedical imaging, geosciences, physics, and surveillance. Hyperspectral sensors look at objects using a vast portion of the electromagnetic spectrum. Certain objects leave unique "fingerprints" in the electromagnetic spectrum. Known as spectral signatures, these "fingerprints" enable identification of the materials that make up a scanned object. For example, a [spectral signature] for oil helps geologists find new [oil field]s.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Sensors

### Sensors
Figuratively speaking, hyperspectral sensors collect information as a set of "images." Each image represents a narrow wavelength range of the electromagnetic spectrum, also known as a spectral band. These "images" are combined to form a three-dimensional (x, y, λ) hyperspectral [data cube] for processing and analysis, where x and y represent two spatial dimensions of the scene, and λ represents the spectral dimension (comprising a range of wavelengths).  
Technically speaking, there are four ways for sensors to sample the hyperspectral cube: spatial scanning, spectral scanning, snapshot imaging, and spatio-spectral scanning.  
Hyperspectral cubes are generated from airborne sensors like NASA's [Airborne Visible/Infrared Imaging Spectrometer] (AVIRIS), or from satellites like NASA's [Earth Observing-1|EO-1] with its hyperspectral instrument Hyperion. However, for many development and validation studies, handheld sensors are used.  
The precision of these sensors is typically measured in spectral resolution, which is the width of each band of the spectrum that is captured. If the scanner detects a large number of fairly narrow frequency bands, it is possible to identify objects even if they are only captured in a handful of pixels. However, [spatial resolution] is a factor in addition to spectral resolution. If the pixels are too large, then multiple objects are captured in the same pixel and become difficult to identify. If the pixels are too small, then the intensity captured by each sensor cell is low, and the decreased [signal-to-noise ratio] reduces the reliability of measured features.  
The acquisition and processing of hyperspectral images is also referred to as [imaging spectroscopy] or, with reference to the hyperspectral cube, as 3D spectroscopy.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Scanning techniques

### Scanning techniques
Photos illustrating individual sensor outputs for the four hyperspectral imaging techniques. From left to right: Slit spectrum; monochromatic spatial map; 'perspective projection' of hyperspectral cube; wavelength-coded spatial map.  
There are four basic techniques for acquiring the three-dimensional (x, y, λ) dataset of a hyperspectral cube. The choice of technique depends on the specific application, seeing that each technique has context-dependent advantages and disadvantages.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Scanning techniques | Spatial scanning

#### Spatial scanning
Acquisition techniques for hyperspectral imaging, visualized as sections of the hyperspectral datacube with its two spatial dimensions (x, y) and one spectral dimension (λ).
In spatial scanning, each two-dimensional (2D) sensor output represents a full slit spectrum (x, λ). Hyperspectral imaging (HSI) devices for spatial scanning obtain slit spectra by projecting a strip of the scene onto a slit and dispersing the slit image with a prism or a grating. These systems have the drawback of having the image analyzed per lines (with a [push broom scanner]) and also having some mechanical parts integrated into the optical train. With these [line-scan camera]s, the spatial dimension is collected through platform movement or scanning. This requires stabilized mounts or accurate pointing information to 'reconstruct' the image. Nonetheless, line-scan systems are particularly common in [remote sensing], where it is sensible to use mobile platforms. Line-scan systems are also used to scan materials moving by on a conveyor belt. A special case of line scanning is point scanning (with a [whisk broom scanner]), where a point-like aperture is used instead of a slit, and the sensor is essentially one-dimensional instead of 2D.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Scanning techniques | Spectral scanning

#### Spectral scanning
In spectral scanning, each 2D sensor output represents a monochromatic (i.e. single wavelength), spatial (x, y)-map of the scene. HSI devices for spectral scanning are typically based on optical band-pass filters (either tunable or fixed). The scene is spectrally scanned by exchanging one filter after another while the platform remains stationary. In such "staring", wavelength scanning systems, spectral smearing can occur if there is movement within the scene, invalidating spectral correlation/detection. Nonetheless, there is the advantage of being able to pick and choose spectral bands, and having a direct representation of the two spatial dimensions of the scene. The most prominent benefits of these [snapshot hyperspectral imaging] systems are the snapshot advantage (higher light throughput) and shorter acquisition time. A number of systems have been designed, including [Computed tomography imaging spectrometer|computed tomographic imaging spectrometry] (CTIS), fiber-reformatting imaging spectrometry (FRIS), [Integral field spectrograph#Lenslet array|integral field spectroscopy with lenslet arrays] (IFS-L), multi-aperture integral field spectrometer (Hyperpixel Array), [Integral field spectrograph#Image slicer|integral field spectroscopy with image slicing mirrors] (IFS-S), image-replicating imaging spectrometry (IRIS), filter stack spectral decomposition (FSSD), coded aperture snapshot spectral imaging (CASSI), image mapping spectrometry (IMS), and multispectral Sagnac interferometry (MSI). However, computational effort and manufacturing costs are high. In an effort to reduce the computational demands and potentially the high cost of non-scanning hyperspectral instrumentation, prototype devices based on [Multivariate optical computing|Multivariate Optical Computing] have been demonstrated. These devices have been based on the [Multivariate optical element|Multivariate Optical Element] spectral calculation engine or the [Spatial light modulator|Spatial Light Modulator] spectral calculation engine. In these platforms, chemical information is calculated in the optical domain prior to imaging such that the chemical image relies on conventional camera systems with no further computing. As a disadvantage of these systems, no spectral information is ever acquired, i.e. only the chemical information, such that post processing or reanalysis is not possible.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Scanning techniques | Spatiospectral scanning

#### Spatiospectral scanning
> Main: Spatiospectral scanning  
In spatiospectral scanning, each 2D sensor output represents a wavelength-coded ("rainbow-colored", λ = λ(y)), spatial (x, y)-map of the scene. A prototype for this technique, introduced in 2014, consists of a camera at some non-zero distance behind a basic slit spectroscope (slit + dispersive element). Advanced spatiospectral scanning systems can be obtained by placing a dispersive element before a spatial scanning system. Scanning can be achieved by moving the whole system relative to the scene, by moving the camera alone, or by moving the slit alone. Spatiospectral scanning unites some advantages of spatial and spectral scanning, thereby alleviating some of their disadvantages.  
Hyperspectral imaging is related to [multispectral imaging]. The distinction between hyper- and multi-band is sometimes based incorrectly on an arbitrary "number of bands" or on the type of measurement. Hyperspectral imaging (HSI) uses continuous and contiguous ranges of wavelengths (e.g. 400 - 1100&nbsp;nm in steps of 1&nbsp;nm) whilst multiband imaging (MSI) uses a subset of targeted wavelengths at chosen locations (e.g. 400 - 1100&nbsp;nm in steps of 20&nbsp;nm).  
Multiband imaging deals with several images at discrete and somewhat narrow bands. Being "discrete and somewhat narrow" is what distinguishes multispectral imaging in the visible wavelength from [color photography]. A multispectral sensor may have many bands covering the spectrum from the visible to the longwave infrared. Multispectral images do not produce the "spectrum" of an object. [Landsat] is a prominent practical example of multispectral imaging.  
Hyperspectral deals with imaging narrow spectral bands over a continuous spectral range, producing the spectra of all pixels in the scene. A sensor with only 20 bands can also be hyperspectral when it covers the range from 500 to 700&nbsp;nm with 20 bands each 10&nbsp;nm wide, while a sensor with 20 discrete bands covering the visible, near, short wave, medium wave and long wave infrared would be considered multispectral.  
In addition, some formal standards define HSI using a minimum number of spectral channels. For example, the IEEE 4001 standard for hyperspectral data characterization defines hyperspectral imagery as containing 30 or more spectral bands, primarily for data interoperability, system classification, and metadata standardization.  
Ultraspectral could be reserved for [interferometer] type imaging sensors with a very fine spectral resolution. These sensors often have (but not necessarily) a low [spatial resolution] of several [pixel]s only, a restriction imposed by the high data rate.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications

### Applications
> See also: Multispectral imaging#Applications  
Hyperspectral remote sensing is used in a wide array of applications. Although originally developed for mining and geology (the ability of hyperspectral imaging to identify various minerals makes it ideal for the mining and oil industries, where it can be used to look for ore and oil), it has now spread into fields as widespread as ecology and surveillance, as well as historical manuscript research, such as the imaging of the [Archimedes Palimpsest]. This technology is continually becoming more available to the public. Organizations such as [NASA] and the [USGS] have catalogues of various minerals and their spectral signatures, and have posted them online to make them readily available for researchers. On a smaller scale, NIR hyperspectral imaging can be used to rapidly monitor the application of pesticides to individual seeds for quality control of the optimum dose and homogeneous coverage.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Zoology

#### Zoology
Hyperspectral imaging is also used in zoology; it is used to investigate the spatial distribution of coloration and its extension into the near-infrared and SWIR range of the spectrum. Some animals for example, such as some tropical frogs and certain leaf-sitting insects are highly reflective in the near-infrared.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Waste sorting and recycling

#### Waste sorting and recycling
Hyperspectral imaging can provide information about the chemical constituents of materials which makes it useful for [waste sorting] and [recycling]. It has been applied to distinguish between substances with different fabrics and to identify natural, animal and synthetic fibers. HSI cameras can be integrated with [machine vision] systems and, via simplifying platforms, allow end-customers to create new waste sorting applications and other sorting/identification applications. A system of [machine learning] and hyperspectral camera can distinguish between 12 different types of plastics such as PET and PP for automated separation of waste of, as of 2020, highly [standardization#Environmental protection|unstandardized] plastics products and [packaging].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Eye care

#### Eye care
Researchers at the [Université de Montréal] are working with [Photon etc.] and Optina Diagnostics to test the use of hyperspectral photography in the diagnosis of [retinopathy] and [macular edema] before damage to the eye occurs.  The metabolic hyperspectral camera will detect a drop in oxygen consumption in the retina, which indicates potential disease. An [ophthalmologist] will then be able to treat the retina with injections to prevent any potential damage.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Mineralogy

#### Mineralogy
A set of stones is scanned with a [[Specim LWIR-C imager in the thermal infrared range from 7.7 μm to 12.4 μm. The [quartz] and [feldspar] spectra are clearly recognizable.  
Hyperspectral remote sensing of minerals is well developed. Many minerals can be identified from airborne images, and their relation to the presence of valuable minerals, such as gold and diamonds, is well understood. Currently, progress is towards understanding the relationship between oil and gas leakages from pipelines and natural wells, and their effects on the vegetation and the spectral signatures. Recent work includes the PhD dissertations of Werff and Noomen.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Surveillance

#### Surveillance
emission measurement, an outdoor scan in winter conditions, ambient temperature -15°C—relative radiance spectra from various targets in the image are shown with arrows. The [Infrared spectroscopy|infrared spectra] of the different objects such as the watch glass have clearly distinctive characteristics. The contrast level indicates the temperature of the object. This image was produced with a [Specim] LWIR hyperspectral imager.  
Traditionally, commercially available thermal infrared hyperspectral imaging systems have needed [liquid nitrogen] or [helium] cooling, which has made them impractical for most surveillance applications. In 2010, [Specim] introduced a thermal infrared hyperspectral camera that can be used for outdoor surveillance and [UAV] applications without an external light source such as the sun or the moon.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Astronomy

#### Astronomy
In astronomy, hyperspectral imaging is used to determine a spatially resolved spectral image. Since a spectrum is an important diagnostic, having a spectrum for each pixel allows more science cases to be addressed. In astronomy, this technique is commonly referred to as [Integral field spectrograph|integral field spectroscopy], and examples of this technique include FLAMES and SINFONI on the [Very Large Telescope]. The [Advanced CCD Imaging Spectrometer] on the [Chandra X-ray Observatory] uses this technique.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Chemical imaging

#### Chemical imaging
> Main: Chemical imaging  
Remote chemical imaging of a simultaneous release of SF6 and NH3 at 1.5 km using the Telops Hyper-Cam imaging spectrometer.
Soldiers can be exposed to a wide variety of chemical hazards. These threats are mostly invisible but detectable by hyperspectral imaging technology. The Telops Hyper-Cam, introduced in 2005, has demonstrated this at distances up to 5&nbsp;km.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Inkjet print analysis

#### Inkjet print analysis  
Hyperspectral imaging in the visible near-infrared (VNIR) spectral range can effectively distinguish between dye-based and pigment-based inkjet prints, despite their similar appearance to the naked eye. Research has demonstrated that these two ink types exhibit significantly different spectral characteristics, particularly in the near-infrared band (800-1000 nm), where dye-based inks show substantially higher reflectance compared to pigment-based inks. This difference is especially pronounced for black ink, where pigment-based ink maintains low reflectance across the entire VNIR spectrum while dye-based ink reflectance increases dramatically in the NIR range. Thus, hyperspectral technology may offer practical applications in detecting print forgery, validating archival-quality prints, and potentially identifying specific printer types used to create documents.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Environment

#### Environment
Top panel: Contour map of the time-averaged spectral radiance at 2078&nbsp;cm−1 corresponding to a CO2 emission line. Bottom panel: Contour map of the spectral radiance at 2580&nbsp;cm−1 corresponding to continuum emission from particulates in the plume. The translucent gray rectangle indicates the position of the stack. The horizontal line at row 12 between columns 64-128 indicate the pixels used to estimate the background spectrum. Measurements made with the Telops Hyper-Cam.
Most countries require continuous monitoring of emissions produced by coal and oil-fired power plants, municipal and hazardous waste incinerators, cement plants, as well as many other types of industrial sources. This monitoring is usually performed using extractive sampling systems coupled with infrared spectroscopy techniques. Some recent standoff measurements performed allowed the evaluation of the air quality but not many remote independent methods allow for low uncertainty measurements.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Civil engineering

#### Civil engineering
Recent research indicates that hyperspectral imaging may be useful to detect the development of cracks in [Pavers (flooring)|pavements] which are hard to detect from images taken with visible spectrum cameras. Furthermore, hyperspectral imaging in the visible, near-infrared, and short-wave infrared spectral ranges can identify pathogens in mixed biofilms, such as Staphylococcus aureus and Pseudomonas aeruginosa, which are often found on surfaces in mono-species and polymicrobial biofilms consisting of combinations of different strains.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Applications | Autonomous vehicles

#### Autonomous vehicles
Hyperspectral imaging has recently been explored for use in autonomous driving and advanced driver-assistance systems. It can be used to improve pedestrian separability and object classification compared with conventional RGB cameras, and to help distinguish road conditions such as water or snow on the surface.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | Advantages and disadvantages

### Advantages and disadvantages
The primary advantage to hyperspectral imaging is that, because an entire spectrum is acquired at each point, the operator needs no prior knowledge of the sample, and postprocessing allows all available information from the dataset to be mined. Hyperspectral imaging can also take advantage of the spatial relationships among the different spectra in a neighbourhood, allowing more elaborate spectral-spatial models for a more accurate [image segmentation|segmentation] and classification of the image.  
The high cost and complexity of hyperspectral imaging equipment has limited its adoption, and analysis of hyperspectral data requires high computing power, especially for real-time or mobile applications. Significant data storage and transmission capacities are necessary, since hyperspectral cubes are large, multidimensional datasets—for example, NASA's [Airborne visible/infrared imaging spectrometer|AVIRIS] sensor, in use since 1987, produces about 500 megabytes per datacube. Significant research has gone into onboard processing of hyperspectral data in satellites, to reduce transmission sizes by only sending detection results. Increasingly, [deep learning] is being researched to lower the computational cost of processing hyperspectral data.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Hyperspectral imaging | See also

### See also  
* [Acousto-optic tunable filter]
* [Airborne real-time cueing hyperspectral enhanced reconnaissance]
* [Cathodoluminescence microscope|Cathodoluminescence]
* [Full spectral imaging]
* [HyMap], a widely used hyperspectral imaging sensor
* [Liquid crystal tunable filter]
* [Metamerism (color)], the perceptual equivalence that hyperspectral imaging overcomes
* [Multispectral image]
* [Sensor fusion]
* [Video spectroscopy]

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Snapshot hyperspectral imaging | Overview

## Snapshot hyperspectral imaging  
### Overview  
Snapshot hyperspectral imaging is a method for capturing hyperspectral images during a single integration time of a detector array. No scanning is involved with this method, in contrast to push broom and whisk broom scanning techniques. The lack of moving parts means that  motion artifacts should be avoided. This instrument typically features detector arrays with a high number of pixels.  
Example of a snapshot hyperspectral imaging spectrometer. The scene is viewed through a lenslet array. Each lenslet transmits the light it receives to the [optic fiber|fiber] to which it is coupled. The bundle of fibers is reformatted and lined up at the entrance slit of a conventional [Diffraction grating|grating] [Optical spectrometer|spectrometer], which [Dispersion (optics)|disperses] the light across the entrance slit onto its detector.
Snapshot hyperspectral imaging is a method for capturing [Hyperspectral imaging|hyperspectral images] during a single integration time of a detector array. No scanning is involved with this method, in contrast to [Push broom scanner|push broom] and [Whisk broom scanner|whisk broom] scanning techniques. The lack of moving parts means that  motion artifacts should be avoided. This instrument typically features detector arrays with a high number of pixels.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Snapshot hyperspectral imaging | Development

### Development
Although the first known reference to a snapshot hyperspectral imaging device—the Bowen "image slicer"—dates from 1938, the concept was not successful until a larger amount of spatial resolution was available. With the arrival of large-format detector arrays in the late 1980s and early 1990s, a series of new snapshot hyperspectral imaging techniques were developed to take advantage of the new technology: a method which uses a [optic fiber|fiber] bundle at the image plane and reformatting the fibers in the opposite end of the bundle to a long line, viewing a scene through a 2D [Diffraction grating|grating] and [Tomographic reconstruction|reconstructing] the multiplexed data with computed [tomography] mathematics, the (lenslet-based) [integral field spectrograph], a modernized version of Bowen's image slicer. More recently, a number of research groups have attempted to advance the technology in order to create devices capable of commercial use. These newer devices include the HyperPixel Array imager a derivative of the integral field spectrograph, a multiaperture spectral filter approach, a [Compressed sensing|compressive-sensing]–based approach using a [coded aperture], a microfaceted-mirror-based approach, a generalization of the [Lyot filter], and a generalization of the [Bayer filter] approach to multispectral filtering.  
[Slitless spectroscopy] can be considered a basic snapshot hyperspectral imaging technique. Spaced point-like sources, such as a sparse field of stars, is a requirement to avoid spectrum overlap on the detector.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Snapshot hyperspectral imaging | Applications

### Applications
Data cube acquired by the [Very Large Telescope].
While snapshot instruments are featured prominently in the research literature, none of these instruments have seen wide adoption in commercial use (i.e. outside the professional [Astronomy|astronomical] community) due to manufacturing limitations. Thus, their primary venue continues to be astronomical [telescope]s. One of the main reasons for the popularity of snapshot devices in the astronomical community is that they offer large increases in the light collection capacity of a telescope when performing hyperspectral imaging. Recent applications have been in soil spectroscopy and vegetation sciences. More recently, snapshot hyperspectral sensors have attracted growing research interest in the context of automotive and autonomous driving applications. Studies suggest that hyperspectral data can enhance perception tasks such as object detection and semantic segmentation by providing material-specific information that is not available in conventional RGB imagery. Comparative evaluations between hyperspectral and RGB modalities indicate potential performance gains in challenging urban scenarios, particularly for pedestrian segmentation and scene understanding.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Overview

## Radar geo-warping  
### Overview  
Radar geo-warping is the adjustment of geo-referenced radar images and video data to be consistent with a geographical projection.  This image warping avoids any restrictions when displaying it together with video from multiple radar sources or with other geographical data including scanned maps and satellite images which may be provided in a particular projection.
There are many areas where geo warping has unique benefits:  
Single radar video signal displayed together with maps of different geographical projections. E.g.
Mercator
UTM
stereographic
Multiple radar video signals displayed simultaneously:
Having the computing power to do so on one computer.
Adapting the projection of all radar signals allowing the geographically correct display and accurate superimposition of those videos.
Slant range correction: a modern 3D radar system can measure the height of a target and hence it is possible to correct the radar video by the real corrected range of the target. Slant Range Correction also allows to compensate the radar tower height e.g. for maritime surveillance radars.  
Radar geo-warping is the adjustment of [Georeferencing|geo-referenced] [radar] images and video data to be consistent with a geographical [map projection|projection].  This [image warping] avoids any restrictions when displaying it together with video from multiple [radar] sources or with other geographical data including scanned maps and [satellite image]s which may be provided in a particular projection.
There are many areas where geo warping has unique benefits:
* Single radar video signal displayed together with maps of different geographical projections. E.g.
**[Mercator projection|Mercator]
**[Transverse Mercator projection|UTM]
**[Stereographic projection|stereographic]
* Multiple radar video signals displayed simultaneously:
** Having the computing power to do so on one computer.
** Adapting the projection of all radar signals allowing the geographically correct display and accurate [superimposition] of those videos.
* [Slant range] correction: a modern [3D radar] system can measure the height of a target and hence it is possible to correct the radar video by the real corrected range of the target. Slant Range Correction also allows to compensate the radar tower height e.g. for maritime surveillance radars.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Introduction

### Introduction
Radar video presents the echoes of electromagnetic waves a radar system has emitted and received as reflections afterwards. These echoes are typically presented on a computer screen with a color-coding scheme depicting the reflection strength.
Two problems have to be solved during such a visualization process. The first problem arises from the fact that typically the radar antenna turns around its position and measures the reflection echo distances from its position in one direction. This effectively means that the radar video data are present in [polar coordinates]. In older systems the polar oriented picture has been displayed in so called [plan position indicator]s (PPI). The PPI-scope uses a radial sweep pivoting about the center of the presentation. This results in a map-like picture of the area covered by the radar beam. A [Phosphor persistence|long-persistence] screen is used so that the display remains visible until the sweep passes again.  
Bearing to the target is indicated by the target's angular position in relation to an imaginary line extending vertically from the sweep origin to the top of the scope. The top of the scope is either true north (when the indicator is operated in the true bearing mode) or ship's heading (when the indicator is operated in the relative bearing mode).
This is a typical plan position indicator (PPI)  
For visualization on a modern computer screen the [Polar coordinate system|polar] coordinates have to be converted into [Cartesian coordinate system|Cartesian] coordinates. This process called radar scan conversion is presented with more detail in the next section.
The second problem to solve arises from the fact that a radar system is placed in the real world and measures real world echo positions. These echoes have to be displayed together with other real world data like object positions, vector maps and satellite images in a consistent way. All this information refers to the curved earth surface but is displayed on a flat computer display. Building a link from real world earth positions to display pixels is commonly called geographical referencing or in short geo-referencing.  
Part of the geo-referencing process is to map the 3D earth surface onto a 2D display. This process of a geographical projection can be performed in many ways, but different data sources have their own 'natural' projection. E.g. Cartesian radar video data from a radar source on the earth surface are geo-referenced by a so-called radar projection. When using this radar projection the Cartesian radar video pixels can directly displayed on a computer screen (only being linearly transformed according to the current position on the screen and e.g. the current zoom level).
A problem now arises if e.g. also a satellite map shall be shown together with the radar video data. The 'natural' geographical projection of a satellite image would be a satellite projection which depends on the satellite orbit, position and further parameters. Now either the satellite image has to be reprojected to a radar projection or the radar video has to use the satellite projection. This geographical re-projection is also called geographical warping or Geo Warping where each image pixel has to be transformed from one projection into another.
This article describes in further detail the Geo Warping of radar video images in real time. It will also show that radar video Geo Warping is done most efficiently when it is integrated with the radar scan conversion process.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Radar-scan conversion

### Radar-scan conversion
This section describes the principles of the radar-scan conversion (RSC) process.
The radar-scan conversion process in general as it is done by the OpenGL RSC
The radar supplies its measured data in polar coordinates (ρ,θ) directly from the rotating antenna. ρ defines the target/echo distance and θ the target angle in polar world coordinates. These data are measured, digitized and stored in a polar coordinate polar store or polar pixmap. The main RSC task is to convert these data to Cartesian (x, y) display coordinates, creating the necessary display pixels. The RSC process is influenced by the current zoom, shift and rotation settings defining which part of the 'world' shall be visible in the display image. As detailed later the RSC process also takes the currently used geographical projection into account when the radar video images are Geo Warped.  
The OpenGL RSC is implemented using a reverse scan conversion approach which calculates for every image pixel the most appropriate radar amplitude value in the polar store. This approach generates an optimal image without any artifacts known from forward spoke fill algorithms. By applying bi-linear filtering between adjacent pixels in the polar store during the conversion process the OpenGL RSC finally achieves a very high visual quality radar display image for every zoom level, creating smooth images of the radar echoes.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Radar projection

### Radar projection
This section illustrates how radar video data are geo referenced and displayed on a computer screen.
This figure shows the principles of a radar measurement
This figure shows an example radar projection with the center of projection (COP) at latitude 50.0° and longitude 0.0° which is also the radar position.  
The radar sensor is positioned on the earth surface with a height h above the ground. It measures the direct distance d to the target (and not e.g. the distance the target is away from the radar if one would move on the earth surface). This distance is then used in the display plane after adjustment to the current display zoom level by the radar scan converter (RSC).
Now it has to be clarified how the radar video data is geo referenced. This basically means, that if we want to display a geographical real world object (like e.g. a light house) which is at the same real world position as the radar target, that it also shall appear at the same position in the display plane. This is realized by calculating the distance from the radar sensor to the respective real world object and use that distance in the display plane. The position of the real world object is typically given in [Geographic coordinate system|geographical coordinates] (latitude, longitude and height above the earth surface).
In other words, using a radar projection with geographical data is done by simulating a radar measurement process with the real world objects and use the resulting range and azimuth in the display plane.  
The second picture to the right shows an example radar projection with the center of projection (COP) at latitude 50.0° and longitude 0.0° which is also the radar position. The dashed lines are the equal-latitude and equal-longitude lines on top of the background map. The solid lines show equal-range and equal-azimuth with the respect to the radar position. It is a feature of the radar projection that equal-range lines are circles and equal-azimuth lines are straight lines. This is necessary to display radar video consistently with other map data when using a radar projection where the projection center has to be the radar position.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Geo Warping process

### Geo Warping process  
Geo Warping Radar to CIB Projection.
This section explains the actual geo warping or re-projection process when applied to radar video in real time.
Assume we want to display radar video on top of a satellite image. As an example we use the CIB projection which is used to display satellite data in CIB [GIS file formats|(Controlled Image Base)] format.  
The Figure Geo Warping Radar to CIB Projection shows dashed the maximal range circle for a range of 111&nbsp;km or 60 miles using the radar projection. Such a range is typical for long range coastal surveillance radars. As stated in the last section this is a perfect circle also on the computer screen. The solid line ellipse shows the same range circle for the CIB projection.  
Typically the errors occurring without Geo Warping are smallest near the radar position if at least the projection center (COP) coincides with the radar position, as realized in our example. Otherwise the error distribution depends both on the used projection and also on the projection parameters. Thus, in our case the errors are most significant near the maximum radar range. The CIB projection error corrected in east–west direction at half the radar range is 2.6&nbsp;km and is 5.3&nbsp;km at the full radar range of 111&nbsp;km. An error of 5.3&nbsp;km is quite significant compared to a typical radial radar measurement resolution of 15&nbsp;m.  
Coordinate re-projection
The Figure Coordinate re-projection explains how the radar coordinates have to be transformed to match the CIB projection coordinates. The radar world coordinates correspond to the Cartesian version of the data measured by the radar sensor. Using an inverse radar projection these coordinates are converted into geographic coordinates which represent the radar data posi-tions on the earth surface. These coordinates are then finally projected by the CIB (or any other) projection for displaying on the computer screen.  
A problem which arises is that geo warping all measured radar video pixels is far too computing resource consuming as to be performed in real time. A possible solution is to use [lookup table]s for all points on the screen, but the lookup table re-computation after e.g. a display zoom operation still causes a noticeable delay for radar video visualization.  
Geo warping grid  
The Figure Geo warping grid depicts the solution to the problem. The circular radar coverage area is divided into a circular grid. Only the corner points of the grid are geo warped which drastically reduces the computation time. Coordinates within a grid tile are computed by a weighted [bilinear interpolation] of the grid corner points.
As geographical projections are typically non-linear functions this introduces a certain error for the radar video display position. Keeping this error sufficiently below the radar measurement resolution makes sure that this is no restriction for the radar video display quality. The grid tile size has to be computed once for a radar position and a given projection. Thus, the grid is typically computed once for a static radar and only more often for moving radars such as on ships.  
The OpenGL radar-scan converter does its scan conversion computations on the [graphics processing unit] to achieve high performance and visual quality. The bi-linear coordinate interpolation mentioned above is done in dedicated hardware on the GPU and therefore causes no overhead for the scan converter.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Radar geo-warping | Example

### Example
This example demonstrates how geo warping helps to consistently display multiple radar videos.  
Example of a radar target shown with and without the effects of geo warping.  
This figure shows the visual effects on the right side without geo warping that targets seen by two radars cannot be correctly displayed and it is unclear where the target is actually positioned. The red and yellow target echoes are seen be radars which are about 50&nbsp;km away. The radars are also about 50&nbsp;km away from each other. The semi-transparent pink color depicts the track history.  
In this scenario even a radar projection is used but of course the radar projection center (COP) can be only at the position of one of the radars. Even larger inconsistencies can arise if a projection different from a radar projection is used. The geo warped view on the left side shows the consistently displayed radar echoes where both radar echoes are exactly at the real target's position.

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Jason satellite series | Overview

## Jason satellite series  
### Overview  
The oceanographic altimeter satellites of the Jason series have been collecting high accuracy measurements of the global ocean's topography since early 1990s, providing a continuous data record of the ocean's response to climate change, especially sea level rise. They also measure topography of inland waters, which is important for understanding droughts and floods. The Jason series satellites have been developed and operated by a broad collaboration of US and European institutions including NASA, CNES, NOAA, EUMETSAT, and ESA. The latest iteration, two identical Jason-CS/Sentinel-6 satellites, is part of the European Union's Copernicus Programme.  
Jason satellites from 1992 to 2016
Copernicus Sentinel-6 by [European Space Agency|ESA]]]
The [:Category: satellites|oceanographic] [altimeter satellite]s of the Jason series have been collecting high accuracy measurements of the [Ocean|global ocean]'s topography since early 1990s, providing a continuous data record of the ocean's response to [climate change], especially [sea level rise]. They also measure topography of inland waters, which is important for understanding [Drought|droughts] and [Flood|floods]. The Jason series satellites have been developed and operated by a broad collaboration of US and European institutions including [NASA], [CNES], [NOAA], [EUMETSAT], and [European Space Agency|ESA]. The latest iteration, two identical Jason-CS/Sentinel-6 satellites, is part of the [European Union]'s [Copernicus Programme].

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Jason satellite series | Satellites

### Satellites
* [TOPEX/Poseidon] (1992–2006), predecessor to the Jason satellites
* [Jason-1] (2001–2013)
* [OSTM/Jason-2] (2008–2019)
* [Jason-3], launched in 2016
* [Sentinel-6 Michael Freilich|Sentinel-6 Michael Freilich] (Jason-CS A: Jason Continuity of Service-A), launched in 2020
* [Sentinel-6B] (Jason-CS B: Jason Continuity of Service-B), launched in November 2025
* [Sentinel-6C], expected to launch in 2030s

Considers that confidence score as the weight  w_k^c  for the feature map  A_k . | Jason satellite series | See also

### See also  
* [Sea state]
* [Surface Water and Ocean Topography|SWOT satellite]
* [List of European Space Agency programmes and missions|List of ESA programmes and missions]
* [PROBA satellite series]

Due to their stochastic nature, a solution is not guaranteed. | Comparison gallery of image scaling algorithms | Scaling methods | [Pixel art scaling algorithms] (hqx)

#### [Pixel art scaling algorithms] (hqx)  
For magnifying computer graphics with low resolution and few colors (usually from 2 to 256 colors), better results will be achieved by pixel art scaling algorithms such as [hqx (algorithm)|hqx] or xbr. These produce sharp edges and maintain high level of detail. Unfortunately due to the standardized size of 218x80 pixels, the "Wiki" image cannot use HQ4x or 4xBRZ to better demonstrate the artifacts they may produce such as row shifting.  
The example images use HQ4x and HQ2x respectively.

Due to their stochastic nature, a solution is not guaranteed. | Comparison gallery of image scaling algorithms | Scaling methods | [Pixel art scaling algorithms] (xbr)

#### [Pixel art scaling algorithms] (xbr)
The xbr family is very useful for creating smooth edges. It will however deform the shape significantly, which in many cases creates a very appealing result. However it will create an effect similar to [posterization] by grouping together local areas into a single colour. It will also remove small details if in-between larger ones which connect together.  
The example images use 4xBRZ and 2xBRZ respectively.  
Image-after-scaling smooth

Due to their stochastic nature, a solution is not guaranteed. | Comparison gallery of image scaling algorithms | Scaling methods | [Pixel art scaling algorithms] (GemCutter)

#### [Pixel art scaling algorithms] (GemCutter)
An adaptable technique which can deliver variable amounts of detail or smoothness. It aims to preserve the shape and coordinates of original details, without blurring those details into neighboring ones. It will avoid blending pixels which directly touch each other, and instead only blend pixels with their diagonal neighbors.  
The "Cutter" name comes from its tendency to cut corners of squares and turn them into diamonds, as well as create distinct faces along stair-stepped pixels, i.e. those which exist on along the angles of edges found on a diamond. The "Gem" prefix both refers to the diamond cut, and also many traditional gem cuts which involve cutting corners at a 45-degree angle.  
The example images use GemCutter Preserve Details (Top), and GemCutter Smooth Edges (Bottom).

Due to their stochastic nature, a solution is not guaranteed. | NAOS (satellite) | Overview

## NAOS (satellite)  
### Overview  
NAOS (National Advanced Optical System) is a high-resolution Earth observation satellite developed by OHB Italia for the Luxembourg Directorate of Defence as part of the Luxembourg Earth Observation System (LUXEOSys). Designed for dual-use governmental and military purposes, NAOS provides very high-resolution optical imagery for applications in defense, security, and humanitarian efforts, supporting organizations such as NATO, European Union, and the United Nations. The satellite was launched on August 26, 2025, aboard a SpaceX Falcon 9 rocket from Vandenberg Space Force Base, California.
NAOS will be operated by LUXEOPs, consortium consisting of RHEA System Luxembourg, LUXSPACE, OHB and RHEA System.  
NAOS (National Advanced Optical System) is a high-resolution Earth observation satellite developed by [OHB|OHB Italia] for the [Luxembourg Directorate of Defence] as part of the Luxembourg Earth Observation System (LUXEOSys). Designed for dual-use governmental and military purposes, NAOS provides very high-resolution optical imagery for applications in defense, security, and humanitarian efforts, supporting organizations such as [North Atlantic Treaty Organisation|NATO], [European Union], and the [United Nations]. The satellite was launched on August 26, 2025, aboard a SpaceX [Falcon 9] rocket from [Vandenberg Space Force Base], California.  
NAOS will be operated by LUXEOPs, consortium consisting of RHEA System Luxembourg, LUXSPACE, OHB and RHEA System.

Due to their stochastic nature, a solution is not guaranteed. | NAOS (satellite) | Orbit & operations

### Orbit & operations
NAOS operates in a sun-synchronous Low Earth orbit at approximately 450&nbsp;km altitude, allowing it to circle the Earth about 15 times per day and achieve global coverage. This orbit enables the satellite to capture more than 100 images daily, with a minimum response time of 17 hours from image request to availability.

Due to their stochastic nature, a solution is not guaranteed. | NAOS (satellite) | Launch

### Launch
Originally scheduled for launch in 2023 aboard an [Arianespace] [Vega-C] rocket, NAOS faced delays due to issues with the launch vehicle. The mission was subsequently reassigned to a SpaceX [Falcon 9 Block 5] rocket. The launch took place on August 26, 2025, at 18:53 UTC from Space Launch Complex 4E (SLC-4E) at Vandenberg Space Force Base, California. The Falcon 9 first stage booster, B1063 completed its 27th flight and successfully returned to Landing Zone 4 (LZ-4) at the launch site. The mission also carried secondary payloads as rideshare, including [Dhruva Space] LEAP-1, [Planet Labs] Pelican-3 and Pelican-4, and [Capella Space] Acadia-6, and [Pixxel] Firefly 4,5,6 satellites.

Due to their stochastic nature, a solution is not guaranteed. | NAOS (satellite) | Significance

### Significance
The NAOS satellite represents Luxembourg's growing investment in space-based capabilities, aligning with the nation's strategic goals in defense and international cooperation. By providing high-resolution imagery to NATO, the EU, and other partners, NAOS enhances Luxembourg's contributions to collective security and global monitoring efforts. The project also underscores the increasing role of space in national security, complementing Luxembourg's other space initiatives, such as the [SES-16|GovSat] program and the [O3b mPOWER] constellation.

Due to their stochastic nature, a solution is not guaranteed. | TROLL (satellite) | Overview

## TROLL (satellite)  
### Overview  
TROLL is a Czech Earth observation and technology demonstration CubeSat-type satellite developed by the Brno-based company TRL Space. The satellite carries a hyperspectral imaging camera and a data processing unit for on-orbit image processing. The satellite was launched on Falcon 9's Transporter 12 mission, together with another Czech satellite SATurnin-1, on 14 January 2025.
The satellite is used for environmental monitoring, agriculture, and security purposes. One of the customers using the satellite's data is the Czech Environmental Inspectorate, part of the country's Ministry of the Environment. The inspectorate is using the data e.g. for detecting illegal landfills and waste dumps in forested areas.  
TROLL is a Czech [Earth observation satellite|Earth observation] and technology demonstration [CubeSat]-type satellite developed by the [Brno]-based company TRL Space. The satellite carries a [hyperspectral imaging] camera and a data processing unit for on-orbit image processing. The satellite was launched on [Falcon 9|Falcon&nbsp;9]'s [List of spaceflight launches in January–March 2025#SpXTransporter12|Transporter 12] mission, together with another Czech satellite [SATurnin-1], on 14 January 2025.  
The satellite is used for [environmental monitoring], agriculture, and security purposes. One of the customers using the satellite's data is the Czech Environmental Inspectorate, part of the country's [Ministry of the Environment (Czech Republic)|Ministry of the Environment]. The inspectorate is using the data e.g. for detecting [Illegal dumping|illegal landfills and waste dumps] in forested areas.

Due to their stochastic nature, a solution is not guaranteed. | Digital image processing | History | Image sensors

#### Image sensors
> Main: Image sensor  
The basis for modern [image sensors] is [metal–oxide–semiconductor] (MOS) technology, invented at Bell Labs between 1955 and 1960, This led to the development of digital [semiconductor] image sensors, including the [charge-coupled device] (CCD) and later the [CMOS sensor]. While researching MOS technology, they realized that an electric charge was the analogy of the magnetic bubble and that it could be stored on a tiny [MOS capacitor]. As it was fairly straightforward to [semiconductor device fabrication|fabricate] a series of MOS capacitors in a row, they connected a suitable voltage to them so that the charge could be stepped along from one to the next.  
The [NMOS logic|NMOS] [active-pixel sensor] (APS) was invented by [Olympus Corporation|Olympus] in Japan during the mid-1980s. This was enabled by advances in MOS [semiconductor device fabrication], with [MOSFET scaling] reaching smaller [List of semiconductor scale examples|micron and then sub-micron] levels. The NMOS APS was fabricated by Tsutomu Nakamura's team at Olympus in 1985. The [CMOS] active-pixel sensor (CMOS sensor) was later developed by [Eric Fossum]'s team at the [NASA] [Jet Propulsion Laboratory] in 1993. By 2007, sales of CMOS sensors had surpassed CCD sensors.  
MOS image sensors are widely used in [optical mouse] technology. The first optical mouse, invented by [Richard F. Lyon] at [Xerox] in 1980, used a [6 μm process|5μm] [NMOS logic|NMOS] [integrated circuit] sensor chip. Since the first commercial optical mouse, the [IntelliMouse] introduced in 1999, most optical mouse devices use CMOS sensors.

Noise and [distortion]s: Imperfections in images due to poor lighting, limited sensors, and file compression can result in unclear images that impact accurate image conversion.

# Noise and [distortion]s: Imperfections in images due to poor lighting, limited sensors, and file compression can result in unclear images that impact accurate image conversion.

Variability in image quality: Variations in image quality and resolution, including blurry images and incomplete details, can hinder uniform processing across a database.

# Variability in image quality: Variations in image quality and resolution, including blurry images and incomplete details, can hinder uniform processing across a database.

[Object detection] and Recognition: Identifying and recognising objects within images, especially in complex scenarios with multiple objects and occlusions, poses a significant challenge.

# [Object detection] and Recognition: Identifying and recognising objects within images, especially in complex scenarios with multiple objects and occlusions, poses a significant challenge.

Computational resource intensity: Accessing adequate computational resources for image processing can be challenging and costly, hindering progress without sufficient resources. | Hydra (satellite constellation) | Overview

## Hydra (satellite constellation)  
### Overview  
Hydra is a satellite constellation of thermal imaging Earth observation small satellites under development by the Spanish company Aistech. It is designed to provide thermal infrared imagery for agriculture, hydrology, environmental risk monitoring, and security applications. In 2023, Hydra has been selected by ESA as one of the Copernicus Contributing Missions (CCMs) providing data to the EU's Copernicus Programme.  
Hydra is a [satellite constellation] of thermal imaging [Earth observation] small satellites under development by the Spanish company Aistech. It is designed to provide [Infrared imaging|thermal infrared imagery] for agriculture, hydrology, environmental risk monitoring, and security applications. In 2023, Hydra has been selected by [European Space Agency|ESA] as one of the [Copernicus Programme#Contributing missions|Copernicus Contributing Missions (CCMs)] providing data to the [European Union|EU]'s [Copernicus Programme].

Computational resource intensity: Accessing adequate computational resources for image processing can be challenging and costly, hindering progress without sufficient resources. | Hydra (satellite constellation) | Satellites

### Satellites
The first satellite of the constellation, the 6U [CubeSat] Hydra 2 was launched on 11 January 2026 aboard [Falcon 9]'s rideshare mission [List of spaceflight launches in January–March 2026#SpXTwilight|Twilight]. Aistech published the first images from Hydra 2 in April 2026. The second satellite, Hydra 3, was launched on Falcon 9's [List of spaceflight launches in April–June 2026#CAS500-2|CAS500-2 rideshare mission] on 3&nbsp;May 2026 at 7:00 UTC.

[Signal processing] | Standard test image | Common test image resolutions

### Common test image resolutions
The standard [Image resolution|resolution] of the images is usually 512×512 or 720×576. Most of these images are available as [Tagged Image File Format|TIFF] files from the [University of Southern California]'s Signal and Image Processing Institute. [Kodak] has released 768×512 images, available as [Portable Network Graphics|PNG]s, that was originally on [Photo CD] with higher resolution,  that are widely used for comparing image compression techniques.

[Signal processing] | Reconnaissance satellite | Overview

## Reconnaissance satellite  
### Overview  
A reconnaissance satellite or intelligence satellite (commonly, although unofficially, referred to as a spy satellite) is an Earth observation satellite or communications satellite deployed for military or intelligence applications.
The first generation type (i.e., Corona and Zenit) took photographs, then ejected canisters of photographic film which would descend back down into Earth's atmosphere. Corona capsules were retrieved in mid-air as they floated down on parachutes. Later, spacecraft had digital imaging systems and downloaded the images via encrypted radio links.
In the United States, most information available about reconnaissance satellites is on programs that existed up to 1972, as this information has been declassified due to its age. Some information about programs before that time is still classified information, and a small amount of information is available on subsequent missions.
A few up-to-date reconnaissance satellite images have been declassified on occasion, or leaked, as in the case of KH-11 photographs which were sent to Jane's Defence Weekly in 1984, or US President Donald Trump tweeting a classified image of the aftermath of a failed test of Iran's Safir rocket in 2019.  
A list of the types of U.S. reconnaissance satellites deployed from 1960 onward
Aerial view of Osama bin Laden's compound in the [Pakistan]i city of [Abbottabad] made by the CIA.
KH-4B Corona satellite
Lacrosse radar spy satellite under construction]]
A model of a German SAR-Lupe reconnaissance satellite inside a Cosmos-3M rocket.
Rhyolite)]]  
A reconnaissance satellite or intelligence satellite (commonly, although unofficially, referred to as a spy satellite) is an [Earth observation satellite] or [communications satellite] deployed for [Military intelligence|military] or [espionage|intelligence] applications.  
The first generation type (i.e., [Corona (satellite)|Corona] and [Zenit (satellite)|Zenit]) took photographs, then ejected canisters of [photographic film] which would descend back down into Earth's atmosphere. Corona capsules were [mid-air retrieval|retrieved in mid-air] as they floated down on [parachute]s. Later, spacecraft had digital imaging systems and downloaded the images via [encrypted] radio links.  
In the United States, most information available about reconnaissance satellites is on programs that existed up to 1972, as this information has been [Declassification|declassified] due to its age. Some information about programs before that time is still [classified information], and a small amount of information is available on subsequent missions.  
A few up-to-date reconnaissance satellite images have been declassified on occasion, or leaked, as in the case of [KH-11] photographs which were sent to [Jane's Defence Weekly] in 1984, or US President [Donald Trump] [Donald Trump on social media|tweeting] a classified image of the aftermath of a failed test of Iran's [Safir (rocket)|Safir] rocket in 2019.

[Signal processing] | Reconnaissance satellite | History

### History
On 16 March 1955, the [United States Air Force] officially ordered the development of an advanced reconnaissance satellite to provide continuous surveillance of "preselected areas of the Earth" in order "to determine the status of a potential enemy's war-making capability".  
During the mid-late 1950s, both the United States and the Soviet Union took interest into reconnaissance satellites. The United States began the [CORONA (satellite)|CORONA] project, which encompassed several series of launches starting in 1959 and ending in 72. This program was made a priority to photograph denied areas, replace the [Lockheed U-2|U-2], and due to public concern about a technological gap between the West and the Soviet Union. It was expedited significantly after the shooting of a U-2 in 1960.  
Meanwhile, in the Soviet Union, a decree that authorized the development of [Sputnik 1|Sputnik] apparently authorized a program for a satellite to be used for photo reconnaissance. This design evolved into Vostok, while another version became Zenit, which was an unmanned reconnaissance satellite. Zenit was launched from 1961 to 1994, however the last flight in 1994 was as a test payload.  
Both the CORONA and Zenit satellites had to be recovered in order to access the used film, making them distinct from future reconnaissance satellites that could transmit photos without returning film to earth.

[Signal processing] | Reconnaissance satellite | Types

### Types
There are several major types of reconnaissance satellite.  
;Missile early warning
> Main: Defense Support Program
> Main: Space-Based Infrared System  
:Provides warning of an attack by detecting [ballistic missile] launches. Earliest known are [Missile Defense Alarm System].  
;Nuclear explosion detection
:[nuclear detonation detection system|Detects nuclear detonation] from space. [Vela (satellite)|Vela] is the earliest known.  
;Electronic reconnaissance
:[Signals intelligence], intercepts stray [radio] waves. [SOLRAD] is the earliest known.  
;Optical imaging surveillance
:[Earth imaging satellite]s. [Satellite images] can be a survey or close-look [Telephoto lens|telephoto]. [Corona (satellite)|Corona] is the earliest known. [Spectral imaging] is commonplace.  
;Radar imaging surveillance
:Most [space-based radar]s use [synthetic-aperture radar]. Can be used at night or through [cloud cover]. Earliest known are the Soviet [US-A] series.

[Signal processing] | Reconnaissance satellite | Missions

### Missions
Examples of reconnaissance satellite missions:
* High resolution photography ([IMINT])
* Measurement and Signature Intelligence ([MASINT])
* Communications eavesdropping ([Signals intelligence|SIGINT])
* Covert communications
* Monitoring of [Comprehensive Test Ban Treaty|nuclear test ban] compliance (see [National Technical Means])
* Detection of missile launches  
On 28 August 2013, it was thought that "a $1-billion high-powered spy satellite capable of snapping pictures detailed enough to distinguish the make and model of an automobile hundreds of miles below" was launched from California's Vandenberg Air Force Base using a Delta IV Heavy launcher, America's highest-payload space launch vehicle at the time.  
On 17 February 2014, a Russian Kosmos-1220 originally launched in 1980 and used for naval missile targeting until 1982, made an uncontrolled [atmospheric entry].

[Signal processing] | Reconnaissance satellite | Benefits

### Benefits
During the 1950s, a Soviet hoax had led to American fears of a [bomber gap]. In 1968, after gaining satellite photography, the United States' intelligence agencies were able to state with certainty that "No new [ICBM] complexes have been established in the USSR during the past year". President [Lyndon B. Johnson] told a gathering in 1967:  
During his [1980 State of the Union Address], President [Jimmy Carter] argued that all of humanity benefited from the presence of American spy satellites:  
Reconnaissance satellites have been used to enforce human rights, through the [Satellite Sentinel Project], which monitors atrocities in [Sudan] and [South Sudan].  
Additionally, companies such as [GeoEye] and [DigitalGlobe] have provided commercial satellite imagery in support of natural disaster response and humanitarian missions.

[Signal processing] | Reconnaissance satellite | In fiction

### In fiction
Spy satellites are commonly seen in [spy fiction] and [military fiction]. Some works of fiction that focus specifically on spy satellites include:
* [Body of Lies (film)|Body of Lies] (2008)
* [Enemy of the State (film)|Enemy of the State] (1998)
* [Ice Station Zebra] (1968)
* [The OMAC Project] (2005)
* [Parmanu: The Story of Pokhran] (2018)
* [Patriot Games] (1987)

[Signal processing] | Reconnaissance satellite | See also

### See also  
* [Aerial reconnaissance]
* [Defense Support Program] (U.S.)
* [European Union Satellite Centre]
* [Information Gathering Satellite] (Japan)
* [List of intelligence gathering disciplines]
* [List of Kosmos satellites]
* [National Reconnaissance Office] (U.S.)
* [Satcom on the Move]

[Signal processing] | Tyche (satellite) | Overview

## Tyche (satellite)  
### Overview  
Tyche is a Britain's first Intelligence, Surveillance and Reconnaissance (ISR) satellite operated by United Kingdom Space Command. It was launched on 16 August 2024 aboard a Falcon 9 rocket as part of SpaceX's Transporter-11 rideshare mission.
The satellite provides optical imagery of the Earth in support of defence intelligence and other government uses.  
Tyche is a Britain's first [Intelligence, surveillance, target acquisition, and reconnaissance#ISR|Intelligence, Surveillance and Reconnaissance] (ISR) satellite operated by [United Kingdom Space Command]. It was launched on 16 August 2024 aboard a [Falcon 9] rocket as part of SpaceX's Transporter-11 rideshare mission.

[Signal processing] | Tyche (satellite) | Background

### Background
Tyche was developed as part of UK plans to expand sovereign space-based intelligence and reconnaissance systems. The satellite was manufactured by [Surrey Satellite Technology] (SSTL).  
Tyche is intended to be the first of four research and development satellites.

[Signal processing] | Tyche (satellite) | Design

### Design
Tyche is a SSTL [Carbonite-2|Carbonite-class] [microsatellite] with a mass of about , carrying a high-resolution optical payload delivering [Ground sample distance|sub-1 metre imagery] of  wide ground areas, including video. Tyche has a planned five-year lifespan orbiting at an altitude of about  in a [sun-synchronous orbit].  
Belgian company Rhea and American [Lockheed Martin] are developing ground-based software to control the ISR satellites.

[Signal processing] | Tyche (satellite) | Mission

### Mission
Tyche provides daytime imagery of the Earth to support military operations, disaster monitoring and related government tasks. The [Defence Intelligence Fusion Centre] at [RAF Wyton] has been involved in processing the imagery generated.

[Signal processing] | Aerial photographic and satellite image interpretation | Overview

## Aerial photographic and satellite image interpretation  
### Overview  
Aerial photographic and satellite image interpretation, or just image interpretation when in context, is the act of examining photographic images, particularly airborne and spaceborne, to identify objects and judging their significance. This is commonly used in military aerial reconnaissance, using photographs taken from reconnaissance aircraft and reconnaissance satellites.
The principles of image interpretation have been developed empirically for more than 150 years.  The most basic are the elements of image interpretation: location, size, shape, shadow, tone/color, texture, pattern, height/depth and site/situation/association.  They are routinely used when interpreting aerial photos and analyzing photo-like images.  An experienced image interpreter uses many of these elements intuitively. However, a beginner may not only have to consciously evaluate an unknown object according to these elements, but also analyze each element's significance in relation to the image's other objects and phenomena.  
U.S. [National Geospatial-Intelligence Agency#National Photographic Interpretation Center (NPIC)|National Photographic Interpretation Center] during the [Cuban Missile Crisis].]]  
Aerial photographic and satellite image interpretation, or just image interpretation when in context, is the act of examining [photographic image]s, particularly [Aerial photography|airborne] and [satellite imagery|spaceborne], to identify objects and judging their significance. This is commonly used in military [aerial reconnaissance], using photographs taken from [reconnaissance aircraft] and [reconnaissance satellite]s.  
The principles of image interpretation have been developed empirically for more than 150 years.  The most basic are the elements of image interpretation: location, size, shape, shadow, tone/color, texture, pattern, height/depth and site/situation/association.  They are routinely used when interpreting aerial photos and analyzing photo-like images.  An experienced image interpreter uses many of these elements intuitively. However, a beginner may not only have to consciously evaluate an unknown object according to these elements, but also analyze each element's significance in relation to the image's other objects and phenomena.

[Signal processing] | Aerial photographic and satellite image interpretation | Angle of view | Vertical imagery and photographs

### Angle of view  
#### Vertical imagery and photographs
Vertical aerial photographs represent more than 95% of all captured aerial images. The principles of capturing vertical photographs are shown in Figure 2. Two major axes which originate from the camera lens are included. Areas in a vertical aerial photograph often have a [Consistency|consistent] size. is unobservable in a low oblique aerial photograph. Black and white aerial photography is capable of producing good-quality images under poor weather conditions, such as foggy and misty air. Color photographs can be used to distinguish different kinds of soils, rocks, and deposits that are located above the rock layers, and some contaminated water sources. This type of photograph is best suited for local [Geotechnical investigation|site investigations]. When the moistened film dries, it expands through one orientation and contracts through the other orientation. Only a small amount of distortion is caused by this.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | SSOT (satellite) | Overview

## SSOT (satellite)  
### Overview  
The Satellite System for Terrestrial Observation, Sistema Satelital para Observación de la Tierra (SSOT), also known as FASat-Charlie, is a Chilean satellite which was launched on December 16, 2011. The objective of the SSOT is to have a satellite system for the observation of Earth based on international cooperation.
The project was commissioned by the Ministry of Defense from the European space manufacturer EADS Astrium - based in Toulouse, France - and had an acquisition cost of 72.5 million dollars, according to the contract signed on July 25, 2008. The Soyuz rocket was used to put the satellite into orbit, which was launched in French Guiana from the spaceport of Kourou, currently used by the European Space Agency.
SSOT is a Miniaturized satellite built on the Myriade satellite bus by Astrium (now Airbus). It was part of a six-satellite payload along with Pléiades-HR 1, ELISA 1, ELISA 2, ELISA 3 and ELISA 4.  
The Satellite System for Terrestrial Observation, Sistema Satelital para Observación de la Tierra (SSOT), also known as FASat-Charlie, is a [Chile|Chilean] [satellite] which was launched on December 16, 2011. The objective of the SSOT is to have a satellite system for the [Earth observation|observation of Earth] based on international cooperation.
The project was commissioned by the [Ministry of National Defense (Chile)|Ministry of Defense] from the European space manufacturer [Astrium|EADS Astrium] - based in [Toulouse], France - and had an acquisition cost of 72.5 million dollars, according to the contract signed on July 25, 2008. The [Soyuz (spacecraft)|Soyuz] rocket was used to put the satellite into orbit, which was launched in French Guiana from the spaceport of [Guiana Space Centre|Kourou], currently used by the [European Space Agency].  
SSOT is a [Miniaturized satellite] built on the [Myriade] [satellite bus] by [Astrium] (now Airbus). It was part of a six-satellite [payload] along with [Pleiades satellites|Pléiades-HR 1], [ELISA 1], [ELISA 2], [ELISA 3] and [ELISA 4].

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | SSOT (satellite) | Background

### Background
Prior to FASat-Charlie, Chile had two experiences with [Small satellite|microsatellites]. The first, [FASat-Alfa], was launched on August 31, 1995. It did not manage to separate from its mother satellite, the Ukrainian [Sich-1], and so the two remain in orbit. The failure was caused by a fault in the pyrotechnic system that allowed the separation and rupture of the spring that joined the two parts, and both are still monitored by [North American Aerospace Defense Command|NORAD]. Three years after the initial failed attempt, FASat-Bravo launched. It became the first artificial Chilean satellite to orbit the Earth independently. In the third year of life, this satellite became inoperative due to power system failures that stopped its batteries from charging, and it became [Space debris|space junk].

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | SSOT (satellite) | Function and Design | Civil Applications

### Function and Design  
#### Civil Applications
According to a report carried out by national specialists, around 180 civil applications have been identified for the satellite relating to agriculture: precision agriculture, forestry, land use planning, mapping of urban areas, growth studies and land use, population dynamics, biomass from forestry, forest cadastres, border protection and monitoring of major works or catastrophes. From the captured images, urban growth, connectivity, tourism, forestry, environmental protection and agriculture can be regulated.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | SSOT (satellite) | Function and Design | Design

#### Design
FASat-Charlie is a small satellite made of [silicon carbide], a material as hard as [sapphire] and less deformed than steel.  
* Mass : 116 kilograms
* Speed : 7.5 kilometres per second .
* Telescope : It has a ground resolution of 1.45 m in panchromatic and 5.8 m in multispectral (visible and NIR spectrum). It delivers around 100 images per day, with a frequency of 3 days in a polar orbit, at a distance of 620 kilometres high.
* Expected operational life : ≥ 5 years

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Geographic information system | Data output and cartography | Web mapping

#### Web mapping
> Main: Web mapping  
There has been a proliferation of free-to-use and easily accessible mapping software such as the [proprietary software|proprietary] web applications [Google&nbsp;Maps] and [Bing&nbsp;Maps], as well as the [free and open-source software|free and open-source] alternative [OpenStreetMap]. These services give the public access to huge amounts of geographic data, perceived by many users to be as trustworthy and usable as professional information. For example, during the COVID-19 pandemic, web maps hosted on dashboards were used to rapidly disseminate case data to the general public.  
Some of them, like Google Maps and [OpenLayers], expose an [application programming interface] (API) that enable users to create custom applications. These toolkits commonly offer street maps, aerial/satellite imagery, geocoding, searches, and routing functionality. Web mapping has also uncovered the potential of [crowdsourcing] geodata in projects like [OpenStreetMap], which is a collaborative project to create a free editable map of the world. These [Mashup (web application hybrid)|mashup] projects have been proven to provide a high level of value and benefit to end users outside that possible through traditional geographic information.  
Web mapping also has drawbacks. Web mapping allows for the creation and distribution of maps by people without proper cartographic training. This has led to maps that ignore cartographic conventions and are potentially misleading, with one study finding that more than half of United States state government COVID-19 dashboards did not follow these conventions.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Geographic information system | Uses | Topics | Coastal and marine science

##### Coastal and marine science
GIS has been applied to the long-term monitoring of shoreline change and coastal erosion, integrating historical aerial photography, satellite imagery and GPS surveys to quantify retreat rates and assess coastal vulnerability.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Overview

## Image resolution  
### Overview  
Image resolution is the level of detail of an image. The term applies to digital images, film images, and other types of images. "Higher resolution" means more image detail.
Image resolution can be measured in various ways. Resolution quantifies how close lines can be to each other and still be visibly resolved. Resolution units can be tied to physical sizes (e.g. lines per mm, lines per inch), to the overall size of a picture (lines per picture height, also known simply as lines, TV lines, or TVL), or to angular subtense. Instead of single lines, line pairs are often used, composed of a dark line and an adjacent light line; for example, a resolution of 10 lines per millimeter means 5 dark lines alternating with 5 light lines, or 5 line pairs per millimeter (5 LP/mm). Photographic lens are most often quoted in line pairs per millimeter.  
Image resolution is the level of detail of an [image]. The term applies to digital images, film images, and other types of images. "Higher resolution" means more image detail.
Image resolution can be measured in various ways. Resolution quantifies how close lines can be to each other and still be visibly resolved. Resolution units can be tied to physical sizes (e.g. lines per mm, lines per inch), to the overall size of a picture (lines per picture height, also known simply as lines, TV lines, or TVL), or to angular subtense. Instead of single lines, line pairs are often used, composed of a dark line and an adjacent light line; for example, a resolution of 10 lines per millimeter means 5 dark lines alternating with 5 light lines, or 5 line pairs per millimeter (5 LP/mm). Photographic lens are most often quoted in line pairs per millimeter.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Types | Pixel count

#### Pixel count
The term resolution is often considered equivalent to [pixel] count in [digital imaging], though international standards in the [digital camera] field specify it should instead be called "Number of Total Pixels" in relation to image sensors, and as "Number of Recorded Pixels" for what is fully captured. Hence, [CIPA DCG-001] calls for notation such as "Number of Recorded Pixels 1000 × 1500". According to the same standards, the "Number of Effective Pixels" that an [image sensor] or [digital camera] has is the count of [pixel] sensors that contribute to the final image (including pixels not in said image but nevertheless support the image filtering process), as opposed to the number of total pixels, which includes unused or light-shielded pixels around the edges.  
An image of N pixels height by M pixels wide can have any resolution less than N lines per picture height, or N TV lines. But when the pixel counts are referred to as "resolution", the convention is to describe the pixel resolution with the set of two positive [integer] numbers, where the first number is the number of pixel columns (width) and the second is the number of pixel rows (height), for example as 7680 × 6876. Another popular convention is to cite resolution as the total number of pixels in the image, typically given as number of [megapixel]s, which can be calculated by multiplying pixel columns by pixel rows and dividing by one million. Other conventions include describing pixels per length unit or pixels per area unit, such as [pixels per inch] or per square inch. None of these pixel resolutions are true resolutions, but they are widely referred to as such; they serve as [upper bound]s on image resolution.  
Below is an illustration of how the same image might appear at different pixel resolutions, if the pixels were poorly rendered as sharp squares (normally, a smooth image reconstruction from pixels would be preferred, but for illustration of pixels, the sharp squares make the point better).  
An image that is 2048 pixels in width and 1536 pixels in height has a total of 2048×1536 = 3,145,728 pixels or 3.1 megapixels.  One could refer to it as 2048 by 1536 or a 3.1-megapixel image. The image would be a very low-quality image (72ppi) if printed at about 28.5 inches wide, but a very good-quality image (300ppi) if printed at about 7 inches wide.  
The number of photodiodes in a color [digital camera] image sensor is often a multiple of the number of pixels in the image it produces, because information from an array of color [image sensors] is used to reconstruct the color of a single pixel. The image has to be interpolated or [demosaic]ed to produce all three colors for each output pixel.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Types | Spatial resolution

#### Spatial resolution
> Main: Spatial resolution  
> Further: Focus (optics)  
The terms blurriness and sharpness are used for digital images, but other descriptors are used to reference the hardware capturing and displaying the images.  
Spatial resolution in radiology is the ability of the imaging modality to differentiate two objects. Low spatial resolution techniques will be unable to differentiate between two objects that are relatively close together.
1951 USAF resolution test target is a classic test target used to determine spatial resolution of imaging sensors and imaging systems.]]  
The measure of how closely lines can be resolved in an image is called spatial resolution, and it depends on properties of the system creating the image, not just the pixel resolution in [pixels per inch] (ppi). For practical purposes, the clarity of the image is decided by its spatial resolution, not the number of pixels in an image. In effect, spatial resolution is the number of independent pixel values per unit length.  
The spatial resolution of consumer displays ranges from 50 to 800 pixel lines per inch.  With scanners, [optical resolution] is sometimes used to distinguish spatial resolution from the number of pixels per inch.  
In [remote sensing], spatial resolution is typically limited by [diffraction limit|diffraction], as well as by aberrations, imperfect focus, and atmospheric distortion. The [ground sample distance] (GSD) of an image, the pixel spacing on the Earth's surface, is typically considerably smaller than the resolvable spot size.  
In [astronomy], one often measures spatial resolution in data points per arcsecond subtended at the point of observation, because the physical distance between objects in the image depends on their distance away and this varies widely with the object of interest.  On the other hand, in [electron microscopy], line or fringe resolution is the minimum separation detectable between adjacent parallel lines (e.g., between planes of atoms), whereas point resolution is instead the minimum separation between adjacent points that can be both detected and interpreted e.g., as adjacent columns of atoms, for instance.  The former often helps one detect periodicity in specimens, whereas the latter (although more difficult to achieve) is key to visualizing how individual atoms interact.  
In Stereoscopic 3D images, spatial resolution could be defined as the spatial information recorded or captured by two viewpoints of a [stereo camera] (left and right camera).

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Types | Spectral resolution

#### Spectral resolution
> Main: ICC profile  
Pixel encoding limits the information stored in a digital image, and the term color profile is used for digital images, but other descriptors are used to reference the hardware capturing and displaying the images.  
Spectral resolution is the ability to resolve spectral features and bands into their separate components. [Color image]s distinguish light of different [visible spectrum|spectra]. [Multispectral image]s can resolve even finer differences of spectrum or [wavelength] by measuring and storing more than the traditional 3 of common RGB color images.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Types | Temporal resolution

#### Temporal resolution
> Main: Frame rate  
Temporal resolution (TR) is the precision of a measurement with respect to time.  
[Movie camera]s and [high-speed camera]s can resolve events at different points in time. The time resolution used for movies is usually 24 to 48 [frames per second] (frames/s), whereas high-speed cameras may resolve 50 to 300 frames/s, or even more.  
The [Heisenberg uncertainty principle] describes the fundamental limit on the maximum spatial resolution of information about a particle's coordinates imposed by the measurement or existence of information regarding its momentum to any degree of precision.  
This fundamental limitation can, in turn, be a factor in the maximum imaging resolution at subatomic scales, as can be encountered using [scanning electron microscope]s.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Types | Radiometric resolution

#### Radiometric resolution
> Main: Color depth  
[Radiometric] resolution determines how finely a system can represent or distinguish differences of [luminous intensity|intensity], and is usually expressed as a number of levels or a number of [bit]s, for example, 8 bits or 256 levels, which is typical of computer image files. The higher the radiometric resolution, the more subtle differences of intensity or [reflectivity] can be represented, at least in theory. In practice, the effective radiometric resolution is typically limited by the noise level, rather than by the number of bits of representation.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Resolution in various media

### Resolution in various media
This is a list of traditional, analogue horizontal resolutions for various media. The list only includes popular formats, not rare formats, and all values are approximate, because the actual quality can vary machine-to-machine or tape-to-tape. For ease-of-comparison, all values are for the NTSC system. (For PAL systems, replace 480 with 576.) Analog formats usually had less chroma resolution.  
Many cameras and displays offset the color components relative to each other or mix up temporal with spatial resolution:  
File:Bayer matrix.svg|[digital camera] (Bayer [color filter array])
File:Lcd display dead pixel.jpg|[LCD] (Triangular [pixel geometry])
File:Shadow mask closeup cursor.jpg|[Cathode-ray tube|CRT] (shadow mask)

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Resolution in various media | Narrowscreen 4:3 computer [display resolution]s

#### Narrowscreen 4:3 computer [display resolution]s
* : [Multi-Color Graphics Array|MCGA]
* : QVGA
* : [Enhanced Graphics Adapter|EGA]
* : [VGA]
* : [Super VGA]
* : XGA / EVGA
* : UXGA

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Resolution in various media | Digital

#### Digital
* : [Video CD]
* : [Digital8]
* : [D-VHS], [DVD], [miniDV], Digital Betacam (NTSC)
* : Widescreen DVD (anamorphic) (NTSC)
* : EDTV (Enhanced Definition Television)
* : [D-VHS], [DVD], [miniDV], [Digital8], Digital Betacam (PAL/SECAM)
*  or : Widescreen DVD (anamorphic) (PAL/SECAM)
* : D-VHS, [HD DVD], [Blu-ray], HDV (miniDV)
* : HDV (miniDV)
* : HDV (miniDV), AVCHD, HD DVD, Blu-ray, HDCAM SR
* : 2K Flat (1.85:1)
* : 2K Digital Cinema
* : QHD (Quad HD) i.e. 4x the pixels in HD 1280x720
* : [4K resolution#Resolutions|4K UHDTV], [Ultra HD Blu-ray]
* : [4K resolution|4K Digital Cinema]
* : [8K resolution|8K UHDTV]
* : [16k resolution|16K Digital Cinema]
* : [32K]
* Sequences from newer films are scanned at 2,000, 4,000, or even 8,000 columns, called [digital cinema|2K, 4K, and 8K], for quality visual-effects editing on computers.
* [IMAX], including IMAX HD and OMNIMAX: approximately  (7,000 lines) resolution. It is about 70&nbsp;MP, which is, , the highest-resolution single-sensor digital cinema camera.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Resolution in various media | Print

#### Print
{| class="wikitable plainrowheaders" style="text-align: right;"  
! scope="col" rowspan=2 | [ISO 216|Paper size]
! scope="col" colspan=2 | Dimensions
! scope="col" rowspan=2 | Pixels  
! scope="col" |
! scope="col" | inches  
! scope="row" | A0  
! scope="row" | A1  
! scope="row" | A2  
! scope="row" | A3  
! scope="row" style="font-weight: bold;" | A4  
! scope="row" | A5  
! scope="row" | A6  
! scope="row" | A7  
! scope="row" | A8

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | Resolution in various media | Modern digital camera resolutions

#### Modern digital camera resolutions
* Digital medium format camera – single, not combined one large digital sensor – 80&nbsp;MP (starting from 2011, current as of 2013) –  or  (81.1&nbsp;MP).
* Digital still camera – [Canon EOS 5DS] – 51&nbsp;MP ()

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Image resolution | See also

### See also
* [Display resolution]
* [Dots per inch]
* [Multi-exposure HDR capture]
* [High-resolution picture transmission]
* [Image scaling]
* [Image scanner]
* [Kell factor], which typically limits the number of visible lines to 0.7x of the device resolution
* [Pixel density]

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Polynomial texture mapping | Overview

## Polynomial texture mapping  
### Overview  
Polynomial texture mapping (PTM), also known as Reflectance Transformation Imaging (RTI), is a technique of imaging and interactively displaying objects under varying lighting conditions to reveal surface phenomena. The data acquisition method is single camera multi light (SCML).  
Polynomial [texture mapping] (PTM), also known as Reflectance Transformation Imaging (RTI), is a technique of [digital imaging|imaging] and [interactive media|interactively] displaying objects under varying [lighting] conditions to reveal surface phenomena. The data acquisition method is [Single camera multi light imaging|single camera multi light] (SCML).

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Polynomial texture mapping | Applications

### Applications
Polynomial texture mapping may be used for detailed recording and documentation, [3D modeling], [edge detection], and to aid the study of [Epigraphy|inscriptions], [rock art] and other artefacts. It has been applied to hundreds of the [Vindolanda tablets] by the [Centre for the Study of Ancient Documents] at the [University of Oxford] in conjunction with the [British Museum]. It has also been deployed, by Ben Altshuler of the [Institute for Digital Archaeology], to scan the [Philae obelisk] at [Kingston Lacy] and the [Parian Chronicle] at the [Ashmolean Museum]; in both cases scans revealed significant, previously illegible text. Method was also used for identifying microscopic worked antler from [Star Carr] and recording ancient rock art in [Armenia].  
A 'dome' supporting twenty-four lights has been used to image paintings in the [National Gallery] and produce polynomial texture maps, providing information on condition phenomena for [conservation and restoration of cultural heritage|conservation] purposes. Studies of the technique at the [National Gallery] and [Tate] concluded that it is an effective tool for documenting changes in the condition of paintings, more easily repeatable than [raking light] photography, and therefore could be used to assess paintings during structural treatment and before and after loan. Twelve dome-based systems built by the University of Southampton have been used to capture thousands of cuneiform tablets at various museums.  
The technique is now also finding uses in the field of [forensic science], for example in imaging footprints, tyre marks, and indented writing.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Dragon (remote sensing) | Overview

## Dragon (remote sensing)  
### Overview  
Dragon is a remote sensing image processing software package. This software provides capabilities for displaying, analyzing, and interpreting digital images from earth satellites and raster data files that represent spatially distributed data.  All the Dragon packages are derived from the code created by Goldin-Rudahl.  
Open Dragon is free to educational users. It was intended to be free worldwide, as well as open source (hence the name) but due to funding problems, it is currently available only in Southeast Asia.
Dragon Academic is functionally identical to Open Dragon.
Dragon Professional is expanded to handle full-scene data sets from sensors such as Landsat TM, SPOT, and Aster.  
Dragon is a [remote sensing] [image processing] software package. This software provides capabilities for displaying, analyzing, and interpreting digital images from earth satellites and raster data files that represent spatially distributed data.  All the Dragon packages are derived from the code created by Goldin-Rudahl.  
* Open Dragon is free to educational users. It was intended to be free worldwide, as well as [open source] (hence the name) but due to funding problems, it is currently available only in Southeast Asia.
* Dragon Academic is functionally identical to Open Dragon.
* Dragon Professional is expanded to handle full-scene data sets from sensors such as [thematic mapper|Landsat TM], [Spot (satellites)|SPOT], and [advanced spaceborne thermal emission and reflection radiometer|Aster].

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Dragon (remote sensing) | History

### History
The initial version of Dragon was released in 1987 and ran on the [MS-DOS]
operating system. Dragon was the first commercial remote sensing software
package designed to use only the native capabilities of off-the-shelf personal
computers. At the time Dragon was developed, other PC remote sensing products
such as [ERDAS IMAGINE|Erdas] required expensive special purpose graphics
devices. Dragon was intended to be used for education in geography, geology,
forestry and other disciplines that use spatial information; thus it was very
important to minimize the costs of required hardware. The first version
of Dragon ran on a basic IBM-PC with two floppy disks and a
four-color or gray-level graphics display. Alternatively, it could use any of several models of Japanese PC.  
The MS-DOS phase of Dragon development focused on trying to squeeze
functionality into very limited disk and memory space, and to get full-color
image display using  rapidly changing graphics
hardware with no standardized drivers. The [VESA] display standard was a
turning point in making full-color display functionality available in
MS-DOS. This VESA/SVGA/MS-DOS version of Dragon can still be adapted
for embedded systems use.  
The move to [Microsoft Windows] 95/98 was painful because these
operating systems did not provide true multitasking. Unfortunately this phase
coincided with the publication of the well-known Gibson and Powers textbook
(Gibson, 2000) which included a copy of the Windows 95 Dragon. With the advent
of Windows NT and successors ([Windows 2000], XP, Vista, etc.), it became
possible to create a Windows version of Dragon that allowed simultaneous
display of and interaction with multiple images.  
In 2004, funding became available from Thailand to create a free educational
version of the software which became known as OpenDragon. This project lasted
for three years.  The software is still available at no cost in Thailand,
Laos, Cambodia and Vietnam (although it has only been translated into Thai).  
After funding for OpenDragon was discontinued, Dragon Professional was
developed to reach beyond the customary educational users. New personal
computer capabilities, which by then extended to gigabytes of memory and
hundreds of gigabytes of disk storage, all at low cost, made it possible to
store and process the very large data sets produced by twenty-first-century
high-resolution satellites.  
Dragon Professional required major changes in the user interaction model,
which previously had assumed a 1-to-1 relationship between the image on the
screen and the sensor data. At the same time, image processing operations such
as selection of ground control points require access to individual data
elements (pixels) selected from the more than 30 million available in a
typical full-scene image. Thus, the appearance and behavior of Dragon
Professional are quite different from OpenDragon/Dragon Academic.

The overlapped areas then appear 3D under the stereoscope.Figure 12: An example of a mirror stereoscope. | Dragon (remote sensing) | The Software

### The Software
Because the expected user is assumed to be relatively untrained, Dragon pays
more attention to the [user experience] than to having a large selection of
possibly obscure processing operations. Within the user interface, which has
been translated into several languages, [context-sensitive help] explains every
user choice, and reasonable defaults are provided where possible. The User
Manual (English only) details all processing algorithms.  
The software provides a fairly conventional set of remote sensing operations,
which are intended to be those which a student of geography arguably ought to
know.  These include:

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Overview

## Noor (satellite)  
### Overview  
Noor (also spelled Nour, Persian: نور, lit.'Light') is a class of Iranian military Earth-imaging CubeSat. Three Noor satellites have been launched from the Shahroud Space Center in Shahrud Desert in Iran into low Earth orbit aboard three-stage Qased (lit. 'message') space-launch vehicles.
Noor-1, the first Iranian military satellite, was launched on 22 April 2020 to a 425 kilometer orbit and decayed from orbit on 13 April 2022 marking a lifespan of one year, eleven months, and nine days, just past its expected one year service life. Noor-1 carried a photo of former Quds Force Commander Qassem Soleimani and a Quranic verse about overcoming adversaries.
Noor-2, the second satellite of the Noor class, was launched on 8 March 2022 (during the Sha'baniyah holiday) to a 500 kilometer orbit. It has a resolution of 12 to 15 meters, a weight of 17 kg, a swath width of 25 km and 6 passes. Noor-2 continues to provide the Islamic Revolutionary Guard Corps with low-resolution overhead imagery.
Noor-3, also called Najm is the third satellite of the Noor class, was launched on a Qassed launcher on 27 September 2023 to a 450 kilometer orbit. It has a weight of 24 kg with a resolution of 6 to 4.8 meters.
According to the Space Commander of the IRGC Aerospace Force, the Noor 3 satellite was stabilized in 1.5 hours and the process it took in the Noor 1 satellite was done automatically in 1 hour in the Noor 3 satellite.The camera used in Noor 3 satellite has up to 2.5 times better photo accuracy than Noor 2. He also added, "In the field of defense, we can use the satellite system for intelligence elites, command and control, and for guiding guided equipment".
The Noor satellite program is a unique development for Iran as it was the first satellite to be developed and launched by the IRGC instead of the Iranian Space Agency.  
Noor (also spelled Nour, ) is a class of [Islamic Revolutionary Guard Corps|Iranian military] [Satellite imagery|Earth-imaging] [CubeSat]. Three Noor [satellite]s have been launched from the [Shahroud Space Center] in [Shahrud, Iran|Shahrud Desert] in [Iran] into [low Earth orbit] aboard [Multistage rocket|three-stage] [Qased (rocket)|Qased] ([Literal translation|lit]. 'message') [Launch vehicle|space-launch vehicles].  
Noor-1, the first Iranian [military satellite], was launched on 22 April 2020 to a 425 kilometer orbit and decayed from [orbit] on 13 April 2022 marking a lifespan of one year, eleven months, and nine days, just past its expected one year [service life]. Noor-1 carried a photo of former [Quds Force] Commander [Qasem Soleimani|Qassem Soleimani] and a Quranic verse about overcoming adversaries.  
[Noor 2 (satellite)|Noor-2], the second satellite of the Noor class, was launched on 8 March 2022 (during the [Mid-Sha'ban|Sha'baniyah holiday]) to a 500 kilometer orbit. It has a resolution of 12 to 15 meters, a weight of 17&nbsp;kg, a swath width of 25&nbsp;km and 6 passes. Noor-2 continues to provide the [Islamic Revolutionary Guard Corps] with [Image resolution|low-resolution] [Satellite imagery|overhead imagery].  
[Noor 3 (satellite)|Noor-3], also called Najm is the third satellite of the Noor class, was launched on a Qassed launcher on 27 September 2023 to a 450 kilometer orbit. It has a weight of 24&nbsp;kg with a resolution of 6 to 4.8 meters.  
According to the Space Commander of the [Islamic Revolutionary Guard Corps Aerospace Force|IRGC Aerospace Force], the Noor 3 satellite was stabilized in 1.5 hours and the process it took in the Noor 1 satellite was done automatically in 1 hour in the Noor 3 satellite.The camera used in Noor 3 satellite has up to 2.5 times better photo accuracy than Noor 2. He also added, "In the field of defense, we can use the satellite system for intelligence elites, command and control, and for guiding guided equipment".  
The Noor satellite program is a unique development for Iran as it was the first satellite to be developed and launched by the [Islamic Revolutionary Guard Corps|IRGC] instead of the [Iranian Space Agency].

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Reaction | Iran

### Reaction  
#### Iran
IRGC Commander-in-Chief General Hossein Salami remarked "Today, the world’s powerful armies do not have a comprehensive defense plan without being in space. Achieving this superior technology, which takes us into space and expands the realm of our abilities, is a strategic achievement."  
Senior [The Pentagon|Pentagon] officials called Iran's satellite launch a provocation. [John E. Hyten|General John Hyten], [Vice Chairman of the Joint Chiefs of Staff|vice chairman of the Joint Chiefs of Staff], stressed on the [Qased (rocket)|Qased] satellite carrier technology, saying that "when you have a missile capable of going a very long way... it means that [Iran] has the ability once again to threaten their neighbors, our allies". The satellite itself, however, was dismissed by [United States Space Force|Space Force] General [John W. Raymond] as "a tumbling [webcam] in space; unlikely providing [Military intelligence|intel]."  
Then [President of the United States|President] [Donald Trump] said that the satellite launch is not an advancement on the Iran's missile program and the showcasing "was only for television," while the US is watching Iran very closely.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Reaction | France

#### France
The [French Foreign Ministry] condemned Iran's launch of a military satellite into orbit. Concurring with the United States' accusations that the same development would contribute to Iran's offensive [ballistic missile] program, the Foreign Ministry said "The Iranian ballistics program is a major concern for regional and [international security]. It contributes to the [Destabilisation|destabilization] of the region and the rise in tensions."

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Reaction | Other

#### Other
[Abdel Bari Atwan], the editor-in-chief of Rai al-Youm and [Al Quds Al Arabi] said that "Iran's recent launched military satellite to space will change the region's equations."

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Controversies

### Controversies  
On 29 July 2020, Iranian state-owned [Fars News Agency] published an article headlined "Al-Udeid Air Base Observed with Noor Satellite" claiming to show an overhead image of the United States' [Al Udeid Air Base] in [Doha], [Qatar] "a gathering place for the [United States Central Command|CENTCOM] [Terrorism|terrorist] [United States Air Force|air force]" imaged by the Noor-1 satellite. Twitter [Open-source intelligence|open-source] commentators suggested that the published image was a "[Palette swap|recolored] [Maxar Technologies|Maxar] image from [Google Earth]".

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Noor (satellite) | Operation

### Operation
Noor satellites circle the Earth once every 90 minutes. The minister's account was banned from Instagram hours later.
{| class="wikitable"  
!English name
![Persian language|Persian] Name
!Launch Date
!Altitude
!Resolution
![Committee on Space Research|COSPAR]
!Operational Status

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Overview

## Satellite imagery in North Korea  
### Overview  
Satellite imagery in North Korea is a knowledge-building tool in the field of North Korean studies. It enables researchers to produce data-based analyses in the agricultural, humanitarian, economic and military fields, in a country where access to the field is limited.  
Satellite image of North Korea in December 2002. Captured by NASA with the Aqua satellite.
[Satellite imagery] in North Korea is a knowledge-building tool in the field of [North Korean studies]. It enables researchers to produce data-based analyses in the agricultural, humanitarian, economic and military fields, in a country where access to the field is limited.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Context

### Context
Collecting data on North Korea is very difficult. States generally produce reliable data, but in North Korea the data produced is often non-existent or of poor quality. When the North Korean government does produce data, its completeness and relevance are often called into question.  
Access to the country is limited and satellite imagery is sometimes the only way to get an overview of important political or military locations. but its use for [North Korean studies] did not emerge until 2012. [Synthetic-aperture radar|SARs] provide a 3D rendering of the earth, even in rainy weather or at night.  
The number of experts in the analysis of satellite imagery in North Korea is limited, has enabled an analysis of the impact on harvests, enabling food needs to be assessed and humanitarian aid to be provided.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Context | Humanitarian aid

#### Humanitarian aid
Some NGOs use satellite images to study the progress of their project as they are not always on the ground.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Context | Military | Nuclear and weapons of mass destruction

#### Military  
##### Nuclear and weapons of mass destruction
> See also: North Korea and weapons of mass destruction  
Analysis of satellite imagery allows us to understand the development of North Korea's nuclear arsenal by observing the infrastructure and activity of nuclear sites. This is sometimes the only way to observe North Korea's nuclear programme, as international and US experts are rarely admitted to the country's nuclear sites.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Context | Human rights

#### Human rights
> See also: Human rights in North Korea  
Satellite imagery is very useful in the field of human rights in the country. The [U.S. Committee for Human Rights in North Korea] has published reports using satellite imagery and defector testimony to analyse the infrastructure and activity of prison camps, particularly to understand renovations, extensions or closures of these sites.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Satellite imagery in North Korea | Satellite imagery producer

### Satellite imagery producer
> See also: Satellite imagery#Imaging satellites  
There are different types of imagery providers. They are military, governmental or commercial
;Military
* United States Government, through the [National Reconnaissance Office] (classified documents)
;Government
* [Landsat program|Landsat] (by the [United States Geological Survey|USGS])
* [European Union], with research programmes (Copernicus...)
;Commercial
* [Maxar Technologies], formerly known as Digital Globe (0.3 m)
* [Airbus DS Geo] (0.5 m)
* [Environmental Systems Research Institute|ESRI]
* [Planet Labs] (imagery captured at daily frequency, 3m)

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Multiple satellite imaging | Overview

## Multiple satellite imaging  
### Overview  
Multiple satellite imaging is the process of using multiple satellites to gather more information than a single satellite so that a better estimate of the desired source is possible. Something that cannot be resolved with one telescope might be visible with two or more telescopes.  
Space Interferometry Mission conceptual picture  
Multiple satellite imaging is the process of using multiple [satellites] to gather more information than a single satellite so that a better estimate of the desired source is possible. Something that cannot be resolved with one [telescope] might be visible with two or more telescopes.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Multiple satellite imaging | Background

### Background
[Interferometry] is the process of combining waves in such a way that they constructively interfere. When two or more independent sources detect a signal at the same given frequency those signals can be combined and the result is better than each one individually. An overview of [Astronomical interferometer]s and a [History of astronomical interferometry] can be referenced from their respective pages.  
The [NASA] [Origins Program] was created in the 1990s to ultimately search for the origin of the [universe]. The theory that the [Origins Program] is based on is: since light travels at a constant speed until it is absorbed by something; there is still light that was part of the first light ever created traveling about the universe and ultimately some of that light is coming in the general direction of Earth. So a satellite system capable of collecting light from the beginning of the [universe] would be able to tell us more about where we came from.  
There is also the constant search for [Astrobiology|life in other worlds]. A satellite system using the interferometric technologies mentioned above would be able to have a much higher resolution than any of the current deep space imaging systems.

The software runs only on Microsoft Windows, although three of its four components also build and run on [Linux]. | Multiple satellite imaging | Future

### Future  
Terrestrial Planet Finder conceptual image by T. Herbst  
[NASA] is currently focused on the [Vision for Space Exploration] and has reduced current funding for scientific unmanned space exploration in favor of human exploration. These budget cuts have slowed the multiple satellite imaging development and relevant scientific missions as [Project Prometheus] and [Terrestrial Planet Finder] have ended as well but research continues.

Mapping of dynamic land features such as wetland plant communities (Jennings et al. 1992) and coastal <br /> land forms (Eleveld et al. 2000);

# Mapping of dynamic land features such as wetland plant communities (Jennings et al. 1992) and coastal <br /> land forms (Eleveld et al. 2000);

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | See also

# Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997).  
### See also
* [SkySat], satellite video

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing software | Overview

## Remote sensing software  
### Overview  
A remote sensing software is a software application that processes remote sensing data. Remote sensing applications are similar to graphics software, but they enable generating geographic information from satellite and airborne sensor data. Remote sensing applications read specialized file formats that contain sensor image data, georeferencing information, and sensor metadata. Some of the more popular remote sensing file formats include: GeoTIFF, NITF, JPEG 2000, ECW (file format), MrSID, HDF, and NetCDF.
Remote sensing applications perform many features including:  
Change Detection — Determining the changes from images taken at different times of the same area
Orthorectification — Warping an image to its location on the earth
Spectral Analysis — For example, using non-visible parts of the electromagnetic spectrum to determine whether a forest is healthy
Image Classification — Categorizing pixels based upon reflectance into different land cover classes (e.g. Supervised classification, Unsupervised classification and Object Oriented classification)
Many remote sensing applications are built using common remote sensing toolkits.  
A remote sensing software is a [software application] that processes [remote sensing] data. Remote sensing applications are similar to [graphics software], but they enable generating [geographic] information from satellite and airborne [sensor] data. Remote sensing applications read specialized file formats that contain sensor image data, georeferencing information, and sensor [metadata]. Some of the more popular remote sensing file formats include: [GeoTIFF], [National Imagery Transmission Format|NITF], [JPEG 2000], [ECW (file format)], [MrSID], [Hierarchical Data Format|HDF], and [NetCDF].  
Remote sensing applications perform many features including:
* Change Detection — Determining the changes from images taken at different times of the same area
* [Orthorectification] — Warping an image to its location on the earth
* Spectral Analysis — For example, using non-visible parts of the [electromagnetic spectrum] to determine whether a forest is healthy
* Image Classification — Categorizing pixels based upon reflectance into different land cover classes (e.g. Supervised classification, Unsupervised classification and Object Oriented classification)  
Many remote sensing applications are built using common remote sensing toolkits.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing software | Examples of remote sensing software

### Examples of remote sensing software
* [PCI Geomatica|Geomatica], PCI Geomatics
* [SAGA GIS] (Open Source)
* [TNTmips], MicroImages
* [ERDAS IMAGINE]
* [ENVI (software)|ENVI]
* [GRASS GIS]
* [OpenEV] (Open Source)
* [Opticks (Software)|Opticks] (Open Source)
* [Orfeo toolbox] (Open Source)
* [RemoteView]
* [SOCET SET]
* [IDRISI]
* [ECognition]
* [ArcGIS]
* [SNAP (remote sensing software)|SNAP]

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing software | See also

### See also
* [Remote sensing]
* [Aerial photography]
* [Geographic information system] (GIS)
* [Radar]
* [Hyperspectral imaging]
* [Image analysis]
* [Multispectral imaging]

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Change detection (GIS) | Overview

## Change detection (GIS)  
### Overview  
In geographic information system (GIS), change detection is a process that measures how the attributes of a particular area have changed between two or more time periods. Change detection often involves comparing aerial photographs or satellite imagery of the area taken at different times. Change detection has been widely used to assess shifting cultivation, deforestation, urban growth, impact of natural disasters like tsunamis, earthquakes, and use/land cover changes etc.  
In [geographic information system] (GIS), change detection is a process that measures how the [attribute (computing)|attributes] of a particular area have changed between two or more time periods. Change detection often involves comparing aerial photographs or satellite imagery of the area taken at different times. Change detection has been widely used to assess shifting cultivation, [deforestation], urban growth, impact of natural disasters like tsunamis, earthquakes, and use/land cover changes etc.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Overview

## Image formation  
### Overview  
The study of image formation encompasses the radiometric and geometric processes by which 2D images of 3D objects are formed.  In the case of digital images, the image formation process also includes analog to digital conversion and sampling.  
The study of image formation encompasses the radiometric and geometric processes by which 2D images of 3D objects are formed.  In the case of [digital image]s, the image formation process also includes analog to digital conversion and [Sampling (signal processing)|sampling].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Imaging

### Imaging  
The imaging process is a mapping of an object to an image plane.  Each point on the image corresponds to a point on the object.  An illuminated object will scatter light toward a lens and the lens will collect and focus the light to create the image.  The ratio of the height of the image to the height of the object is the magnification.  The spatial extent of the image surface and the focal length of the lens determines the field of view of the lens. Image formation of mirror these have a center of curvature and its focal length of the mirror is half of the center of curvature.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Illumination

### Illumination  
An object may be illuminated by the light from an emitting source such as the sun, a light bulb or a Light Emitting Diode. The light incident on the object is reflected in a manner dependent on the surface properties of the object. For rough surfaces, the reflected light is scattered in a manner described by the Bi-directional Reflectance Distribution Function ([Bidirectional reflectance distribution function|BRDF]) of the surface.  The BRDF of a surface is the ratio of the exiting power per square meter per [steradian] ([radiance]) to the incident power per square meter ([irradiance]).  The BRDF typically varies with angle and may vary with wavelength, but a specific important case is a surface that has constant BRDF.  This surface type is referred to as [Lambertian reflectance|Lambertian] and the magnitude of the BRDF is R/π, where R is the reflectivity of the surface.  The portion of scattered light that propagates toward the lens is collected by the [entrance pupil] of the imaging lens over the field of view.  
frameless

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Field of view and imagery

### Field of view and imagery  
The Field of view of a lens is limited by the size of the image plane and the focal length of the lens.  The relationship between a location on the image and a location on the object is y = f*tan(θ), where y is the max extent of the image plane, f is the focal length of the lens and θ is the field of view.  If y is the max radial size of the image then θ is the field of view of the lens.  While the image created by a lens is continuous, it can be modeled as a set of discrete field points, each representing a point on the object.  The quality of the image is limited by the aberrations in the lens and the diffraction created by the finite aperture stop.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Pupils and stops

### Pupils and stops  
The aperture stop of a lens is a mechanical aperture which limits the light collection for each field point.  The entrance pupil is the image of the aperture stop created by the optical elements on the object side of the lens.  The light scattered by an object is collected by the entrance pupil and focused onto the image plane via a series of refractive elements. The cone of the focused light at the image plane is set by the size of the entrance pupil and the focal length of the lens.  This is often referred to as the f-stop or f-number of the lens. f/# = f/D where D is the diameter of the entrance pupil.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Pixelation and color vs. monochrome

### Pixelation and color vs. monochrome  
In typical digital imaging systems, a sensor is placed at the image plane.  The light is focused on to the sensor and the continuous image is pixelated.  The light incident on each pixel in the sensor will be integrated within the pixel and a proportional electronic signal will be generated.  The angular geometric resolution of a pixel is given by atan(p/f), where p is the pitch of the pixel.  This is also called the pixel field of view.  The sensor may be monochrome or color.  In the case of a monochrome sensor, the light incident on each pixel is integrated and the resulting image is a grayscale like picture. For color images, a [color filter array|mosaic color filter] is typically placed over the pixels to create a color image.  An example is a [Bayer filter].  The signal incident on each pixel is then digitized to a bit stream.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Image quality

### Image quality  
The quality of an image is dependent upon both geometric and physical items.  Geometrically, higher density of pixels across an image will give less blocky pixelation and thus a better geometric image quality.  Lens aberrations also contribute to the quality of the image. Physically, diffraction due to the aperture stop will limit the resolvable spatial frequencies as a function of f-number.  
In the [frequency domain], Modulation Transfer Function ([Optical transfer function|MTF]) is a measure of the quality of the imaging system.  The MTF is a measure of the visibility of a sinusoidal variation in irradiance on the image plane as a function of the frequency of the sinusoid.  It includes the effects of diffraction, aberrations and pixelation.  For the lens, the MTF is the autocorrelation of the pupil function, so it accounts for the finite pupil extent and the lens aberrations.  The sensor MTF is the Fourier Transform of the [pixel geometry].  For a square pixel, MTF(ξ) = sin(πξp)/πξp where p is the pixel width and ξ is the spatial frequency.  The MTF of the combination of the lens and detector is the product of the two component MTFs.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Perception

### Perception  
Color images can be perceived via two means.  In the case of computer vision the light incident on the sensor comprises the image.  In the case of visual perception, the human eye has a color dependent response to light so this must be accounted for.  This is important consideration when converting to [grayscale].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image formation | Image formation in eye

### Image formation in eye  
The principal difference between the lens of the eye and an ordinary optical lens is that the former is flexible. The radius of the curvature of the anterior surface of the lens is greater than the radius of its posterior surface. The shape of the lens is controlled by tension in the fibers of the [ciliary body]. To focus on distant objects, the controlling muscles cause the lens to be relatively flattened. Similarly, these muscles allow the lens to become thicker in order to focus on objects near the eye.  
The distance between the center of the lens and the retina ([focal length]) varies from approximately 17&nbsp;mm to about 14&nbsp;mm, as the refractive power of the lens increases from its minimum to its maximum. When the eye focuses on an object farther away than about 3 m, the lens exhibits its lowest refractive power. When the eye focuses on a close object, the lens is most strongly refractive.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral Unit for Land Assessment | Overview

## Multispectral Unit for Land Assessment  
### Overview  
The Multispectral Unit for Land Assessment (MULA) is a planned Filipino satellite dedicated in Earth observation and remote sensing. Upon completion it will become the largest satellite made by Filipinos.  
The Multispectral Unit for Land Assessment (MULA) is a planned Filipino [satellite] dedicated in Earth observation and remote sensing. Upon completion it will become the largest satellite made by Filipinos.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral Unit for Land Assessment | Development

### Development
The [Philippine Space Agency] (PhilSA) started the Multispectral Unit for Land Assessment (MULA) project in 2020 under its Advanced Satellite Program (ASP). The preliminary mission objectives of MULA was determined.  
PhilSA announced on June 9, 2021, that a satellite is in development that would be bigger than the ones made previously under the [Philippine Scientific Earth Observation Microsatellite program|Philippine Scientific Earth Observation Microsatellite (PHL-Microsat) program].  
MULA would be the first of a "next-generation satellites" under the [Philippine space program], with the team behind the satellite building on the knowledge gained in developing the Diwata and Maya nanosatellites. The investment cost for the satellite is at least US$34 million.  
The satellite project is led by John Leur Labrador and is part of the ASP of the [Department of Science and Technology (Philippines)|Department of Science and Technology] (DOST). The [University of the Philippines Diliman] and DOST-[Advanced Science and Technology Institute], in coordination of PhilSA, are the lead entities responsible for MULA's development. It is also co-designed with British firm [Surrey Satellite Technology]. Filipino engineers who worked on MULA were sent to the United Kingdom for an immersion on satellite design and manufacturing process.  
Development was hampered by the [COVID-19 pandemic]. Progress continued from 2023. By June 2025, the MULA project is already in the testing phase.  
At the ninth Philippine Space Council meeting in June 2026, President [Bongbong Marcos] approve plans to develop a [satellite constellation] under the MULA program.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral Unit for Land Assessment | Instruments

### Instruments
MULA will weigh , and will become the largest Filipino-made satellite. It is equipped with a TrueColour camera which has a capability to capture images with a  resolution and a wide swatch width of . MULA will also have nine [spectral bands] for various environmental applications including land cove change mapping, crop monitoring, and disaster and forestry management. It will be designed to be able to take images of roughly  of land area daily.  
It will also be equipped with [Automatic Identification System] (AIS) and [Automatic Dependent Surveillance–Broadcast] (ADS–B) which could be used to detect and track aircraft and ships. The satellite will also have a jet propulsion system.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral Unit for Land Assessment | Launch and mission

### Launch and mission
It was originally planned that MULA would be launched to space by 2023 but this schedule has been postponed to 2025. MULA will be positioned in a sun-synchronous [low Earth orbit], and will rotate around the globe ten times daily.  
During the 8th Philippine Space Council (PSC) meeting held in 12 August 2024, President [Bongbong Marcos] announced that MULA will be launched on a [Falcon 9] rocket as part of [SpaceX] Transporter-16 mission, scheduled for NET February 2026. , the projected launch date is slated for April 2027.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | Overview

## Image rectification  
### Overview  
Image rectification is a transformation process used to project images onto a common image plane. This process has several degrees of freedom and there are many strategies for transforming images to the common plane. Image rectification is used in computer stereo vision to simplify the problem of finding matching points between images (i.e. the correspondence problem), and in geographic information systems (GIS) to merge images taken from multiple perspectives into a common map coordinate system.  
A camera (red) rotates about the blue axis by 5° to 90° (green), as the images are rectified by projection to the virtual image plane (blue). The virtual plane must be parallel to the stereo baseline (orange) and for visualization is located in the center of rotation. In this case, rectification is achieved by a virtual rotation of the red and green image planes, respectively, to be parallel to the stereo baseline.  
Image rectification is a transformation process used to project images onto a common image plane. This process has several degrees of freedom and there are many strategies for transforming images to the common plane. Image rectification is used in [computer stereo vision] to simplify the problem of finding matching points between images (i.e. the [correspondence problem]), and in [geographic information system]s (GIS) to merge images taken from multiple perspectives into a common map coordinate system.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In computer vision

### In computer vision
right  
[Computer stereo vision] takes two or more images with known relative camera positions that show an object from different viewpoints. For each pixel it then determines the corresponding scene point's depth (i.e. distance from the camera) by first finding matching pixels (i.e. pixels showing the same scene point) in the other image(s) and then applying [triangulation (computer vision)|triangulation] to the found matches to determine their depth.
Finding matches in stereo vision is restricted by [epipolar geometry]: Each pixel's match in another image can only be found on a line called the epipolar line.
If two images are coplanar, i.e. they were taken such that the right camera is only offset horizontally compared to the left camera (not being moved towards the object or rotated), then each pixel's epipolar line is horizontal and at the same vertical position as that pixel. However, in general settings (the camera does move towards the object or rotate) the epipolar lines are slanted. Image rectification warps both images such that they appear as if they have been taken with only a horizontal displacement and as a consequence all epipolar lines are horizontal, which slightly simplifies the stereo matching process. Note however, that rectification does not fundamentally change the stereo matching process: It searches on lines, slanted ones before and horizontal ones after rectification.  
Image rectification is also an equivalent (and more often used) alternative to perfect camera coplanarity. Even with high-precision equipment, image rectification is usually performed because it may be impractical to maintain perfect coplanarity between cameras.  
Image rectification can only be performed with two images at a time and simultaneous rectification of more than two images is generally impossible.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In computer vision | Transformation

#### Transformation
> Further: Transformation matrix  
If the images to be rectified are taken from camera pairs without geometric [Distortion (optics)|distortion], this calculation can easily be made with a [linear transformation]. X & Y rotation puts the images on the same plane, scaling makes the image frames be the same size and Z rotation & skew adjustments make the image pixel rows directly line up. The rigid alignment of the cameras needs to be known (by calibration) and the calibration coefficients are used by the transform.  
In performing the transform, if the cameras themselves are calibrated for internal parameters, an [essential matrix] provides the relationship between the cameras. The more general case (without camera calibration) is represented by the [fundamental matrix (computer vision)|fundamental matrix].  If the fundamental matrix is not known, it is necessary to find preliminary point correspondences between stereo images to facilitate its extraction.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In computer vision | Algorithms

#### Algorithms
There are three main categories for image rectification algorithms: planar rectification, cylindrical rectification and polar rectification.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In computer vision | Implementation details

#### Implementation details
All rectified images satisfy the following two properties:
* All epipolar lines are parallel to the horizontal axis.
* Corresponding points have identical vertical coordinates.  
In order to transform the original image pair into a rectified image pair, it is necessary to find a [projective transformation] H. Constraints are placed on H to satisfy the two properties above. For example, constraining the epipolar lines to be parallel with the horizontal axis means that epipoles must be mapped to the infinite point [1,0,0]T in [homogeneous coordinates]. Even with these constraints, H still has four degrees of freedom. It is also necessary to find a matching H'  to rectify the second image of an image pair. Poor choices of H and H'  can result in rectified images that are dramatically changed in scale or severely distorted.  
There are many different strategies for choosing a projective transform H for each image from all possible solutions. One advanced method is minimizing the disparity or least-square difference of corresponding points on the horizontal axis of the rectified image pair. Another method is separating H into a specialized projective transform, similarity transform, and shearing transform to minimize image distortion. One simple method is to rotate both images to look perpendicular to the line joining their collective optical centers, twist the optical axes so the horizontal axis of each image points in the direction of the other image's optical center, and finally scale the smaller image to match for line-to-line correspondence. This process is demonstrated in the following example.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In computer vision | Example

#### Example
Model used for image rectification example
3D view of example scene. The first camera's optical center and image plane are represented by the green circle and square respectively. The second camera has similar red representations.
Set of 2D images from example. The original images are taken from different perspectives (row 1). Using systematic transformations from the example (rows 2 and 3), we are able to transform both images such that corresponding points are on the same horizontal scan lines (row 4).  
Our model for this example is based on a pair of images that observe a 3D point P, which corresponds to p and p'  in the pixel coordinates of each image. O and O'  represent the optical centers of each camera, with known camera matrices  M=K[I~ 0]  and  M'=K'[R~ T]  (we assume the world origin is at the first camera). We will briefly outline and depict the results for a simple approach to find a H and H'  projective transformation that rectify the image pair from the example scene.  
First, we compute the epipoles, e and e'  in each image:
:
e=M \begin{bmatrix} O' \\ 1 \end{bmatrix}
=M\begin{bmatrix} -R^T T \\ 1 \end{bmatrix} = K[I~ 0]\begin{bmatrix} -R^T T \\ 1 \end{bmatrix} = -KR^T T  
:
e'=M'\begin{bmatrix} O \\ 1 \end{bmatrix} = M'\begin{bmatrix} 0 \\ 1 \end{bmatrix} = K'[R~T]\begin{bmatrix} 0 \\ 1 \end{bmatrix} = K'T  
Second, we find a projective transformation H1 that rotates our first image to be parallel to the baseline connecting O and O'  (row 2, column 1 of 2D image set). This rotation can be found by using the [cross product] between the original and the desired optical axes. Next, we find the projective transformation H2 that takes the rotated image and twists it so that the horizontal axis aligns with the baseline. If calculated correctly, this second transformation should map the e to infinity on the x axis (row 3, column 1 of 2D image set). Finally, define  H=H_2H_1  as the projective transformation for rectifying the first image.  
Third, through an equivalent operation, we can find H'  to rectify the second image (column 2 of 2D image set). Note that H'1 should rotate the second image's [optical axis] to be parallel with the transformed optical axis of the first image. One strategy is to pick a plane parallel to the line where the two original optical axes intersect to minimize distortion from the reprojection process. In this example, we simply define H'  using the rotation matrix R and initial projective transformation H as  H' = HR^T .  
Finally, we scale both images to the same approximate resolution and align the now horizontal epipoles for easier horizontal scanning for correspondences (row 4 of 2D image set).  
Note that it is possible to perform this and similar algorithms without having the camera parameter matrices M and M' . All that is required is a set of seven or more image to image correspondences to compute the fundamental matrices and epipoles.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | In geographic information system

### In geographic information system
> Main: Georeferencing  
Image rectification in [Geographic information system#Raster-to-vector translation|GIS] converts images to a [standard map] coordinate system. This is done by matching ground control points (GCP) in the mapping system to points in the image. These GCPs  calculate necessary image transforms.  
Primary difficulties in the process occur
*when the accuracy of the map points are not well known
*when the images lack clearly identifiable points to correspond to the maps.  
The maps that are used with rectified images are non-topographical. However, the images to be used may contain distortion from terrain. Image orthorectification additionally removes these effects.  
Image rectification is a standard feature available with GIS software packages.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image rectification | See also

### See also  
*[Binocular disparity]
*[Correspondence problem]
*[Epipolar geometry]
*[Geographic information system]
*[Georeferencing]
*[Homography]
*[Image registration]
*[Photogrammetry]
*[Point set registration]
*[Rubbersheeting]
*[Distortion (optics)#Software correction|Software correction of lens distortion]
*[Stereo camera]
*[Stereo vision]
*[Structure from motion]

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Overview

## Mobile mapping  
### Overview  
Mobile mapping is the process of collecting geospatial data from a mobile vehicle, typically fitted with a range of GNSS, photographic, radar, laser, LiDAR or any number of remote sensing systems. Such systems are composed of an integrated array of time synchronised navigation sensors and imaging sensors mounted on a mobile platform. The primary output from such systems include GIS data, digital maps, and georeferenced images and video.  
Google Street View Car  
Mobile mapping is the process of collecting [geospatial] data from a mobile [vehicle], typically fitted with a range of [GNSS], [photographic], [radar], [laser scanning|laser], [LiDAR] or any number of [remote sensing] systems. Such systems are composed of an integrated array of time synchronised [navigation] sensors and [Photography|imaging] sensors mounted on a mobile platform. The primary output from such systems include [GIS] data, digital [map]s, and [georeference]d images and video.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | History

### History  
The development of direct reading [georeference|georeferencing] technologies opened the way for mobile mapping systems. [GPS] and [Inertial Navigation System]s, have allowed rapid and accurate determination of [Location (geography)|position] and [:wikt:attitude|attitude] of [remote sensing] equipment, effectively leading to direct mapping of features of interest without the need for complex post-processing of observed data.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Aerial mobile mapping

### Applications  
#### Aerial mobile mapping  
Traditional [Topographic surveying and mapping|techniques] of geo-referencing aerial photography, [ground profiling radar], or [Lidar] are prohibitively expensive, particularly in inaccessible areas, or where the type of data collected makes interpretation of individual features difficult. Image direct georeferencing, simplifies the mapping control for large scale mapping tasks.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Emergency response planning

#### Emergency response planning  
Mobile mapping systems allow rapid collection of data to allow accurate assessment of conditions on the ground.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Internet applications

#### Internet applications  
Internet, and [mobile device] users, are increasingly utilising geo-spatial information, either in the form of mapping, or geo-referenced imaging. Google, Microsoft, and Yahoo have adapted both aerial photographs and satellite images to develop online mapping systems. [Google Street View|Street View] type images are also an increasing market.  
Location aware [Personal digital assistant|PDA] systems rely on geo-referenced features collated from mobile mapping sources.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Road mapping and highway facility management

#### Road mapping and highway facility management  
GPS combined with digital camera systems allow rapid update of road maps.  
The same system can be utilised to carry out efficient road condition surveys, and facilities management. [Laser scanning] technologies, applied in the mobile mapping sense, allow full 3D data collection of slope, bankings, etc.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Road Inventory and Asset Management

#### Road Inventory and Asset Management
Mobile LiDAR with a digital imaging system is being used to gather data which after post-processing generates strip plan, horizontal and vertical profile, all other asset within and beyond ROW including abutting land use and deficient geometry. This also calls for riding quality of pavement, Existing Traffic Characteristics and capacity of the corridor, Speed-flow-density analysis, Road Safety Review of the Corridor, Junction, and median opening, Facilities for commercial vehicles. Thus all data being used to form a performance matrix help identifying the gaps in corridor efficiency for prioritization of interventions to improve corridor efficiency.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Mobile mapping | Applications | Digital Twins applications

#### Digital Twins applications
Mobile mapping combined with indoor mapping are being used in creation of digital twins. These digital twins can be a single building or an entire city or country. Several mobile mapping companies, known as "Maker of Digital Twins" are embarking on capturing the digital twins market amid the growing trend among organizations and governments that are adopting digital twins for Internet of Things and Artificial Intelligence applications within the Industrial Revolution 4.0 framework.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Overview

## Photogrammetry  
### Overview  
Photogrammetry is the science and technology of obtaining reliable information about physical objects and the environment through the process of recording, measuring and interpreting photographic images and patterns of electromagnetic radiant imagery and other phenomena.  
While the invention of the method is attributed to Aimé Laussedat, the term "photogrammetry" was coined by the German architect Albrecht Meydenbauer, which appeared in his 1867 article "Die Photometrographie."  
There are many variants of photogrammetry.  One example is the extraction of three-dimensional measurements from two-dimensional data (i.e. images); for example, the distance between two points that lie on a plane parallel to the photographic image plane can be determined by measuring their distance on the image, if the scale of the image is known.  Another is the extraction of accurate color ranges and values representing such quantities as albedo, specular reflection, metallicity, or ambient occlusion from photographs of materials for the purposes of physically based rendering.
Close-range photogrammetry refers to the collection of photography from a lesser distance than traditional aerial (or orbital) photogrammetry. Photogrammetric analysis may be applied to one photograph, or may use high-speed photography and remote sensing to detect, measure and record complex 2D and 3D motion fields by feeding measurements and imagery analysis into computational models in an attempt to successively estimate, with increasing accuracy, the actual, 3D relative motions.
From its beginning with the stereoplotters used to plot contour lines on topographic maps, it now has a very wide range of uses such as sonar, radar, and lidar.  
Low altitude aerial photograph for use in photogrammetry. Location: Three Arch Bay, [Laguna Beach, California].
Photogrammetry is the science and technology of obtaining reliable information about physical objects and the environment through the process of recording, measuring and interpreting photographic images and patterns of electromagnetic radiant imagery and other phenomena.
Photogrammetry of the headquarters of Fazenda do Pinhal, São Carlos-SP, Brazil
While the invention of the method is attributed to [Aimé Laussedat], the term "photogrammetry" was coined by the German architect , which appeared in his 1867 article "Die Photometrographie."
Photogrammetry of the headquarters of Fazenda do Pinhal, São Carlos-SP, Brazil
There are many variants of photogrammetry.  One example is the extraction of three-dimensional measurements from two-dimensional data (i.e. images); for example, the distance between two points that lie on a plane parallel to the photographic [image plane] can be determined by measuring their distance on the image, if the [scale (map)|scale] of the image is known.  Another is the extraction of accurate [color] ranges and values representing such quantities as [albedo], [specular reflection], [Metallicity#Photometric colors|metallicity], or [ambient occlusion] from photographs of materials for the purposes of [physically based rendering].  
Close-range photogrammetry refers to the collection of photography from a lesser distance than traditional aerial (or orbital) photogrammetry. Photogrammetric analysis may be applied to one photograph, or may use [high-speed photography] and [remote sensing] to detect, measure and record complex 2D and 3D [motion field]s by feeding measurements and [imagery analysis] into [Computer simulation|computational models] in an attempt to successively estimate, with increasing accuracy, the actual, 3D relative motions.  
From its beginning with the [stereoplotter]s used to plot [contour line]s on [topographic map]s, it now has a very wide range of uses such as [sonar], [radar], and [lidar].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Methods

### Methods
A data model of photogrammetry
Tuure Leppänen, Reconstruction I: 2D image from a 3D model built with photogrammetry methods from hundreds of ground-level photos of a japanese garden  
Photogrammetry uses methods from many disciplines, including [optics] and [projective geometry]. Digital image capturing and photogrammetric processing includes several well defined stages, which allow the generation of 2D or 3D digital models of the object as an end product. The data model on the right shows what type of information can go into and come out of photogrammetric methods.  
The 3D coordinates define the locations of object points in the [Three-dimensional space|3D space]. The image coordinates define the locations of the object points' images on the film or an electronic imaging device. The [Extrinsic parameters|exterior orientation] of a camera defines its location in space and its view direction. The [Intrinsic parameters|inner orientation] defines the geometric parameters of the imaging process. This is primarily the focal length of the lens, but can also include the description of lens distortions. Further additional observations play an important role: With scale bars, basically a known distance of two points in space, or known fix points, the connection to the basic measuring units is created.  
Each of the four main variables can be an input or an output of a photogrammetric method.  
Algorithms for photogrammetry typically attempt to minimize the sum of the [Least squares|squares of errors] over the coordinates and relative displacements of the reference points. This minimization is known as [bundle adjustment] and is often performed using the [Levenberg–Marquardt algorithm].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Methods | Stereophotogrammetry

#### Stereophotogrammetry  
> Further: 3D reconstruction from multiple images  
> See also: Computer stereo vision  
A special case, called stereophotogrammetry, involves estimating the three-dimensional [Coordinate system|coordinates] of points on an object employing measurements made in two or more photographic images taken from different positions (see [stereoscopy]). Common points are identified on each image. A line of sight (or ray) can be constructed from the camera location to the point on the object.  It is the intersection of these rays ([triangulation (computer vision)|triangulation]) that determines the three-dimensional location of the point. More sophisticated [algorithm]s can exploit other information about the scene that is known [A priori and a posteriori|a priori], for example [Symmetry|symmetries], in some cases allowing reconstructions of 3D coordinates from only one camera position. Stereophotogrammetry is emerging as a robust non-contacting measurement technique to determine dynamic characteristics and mode shapes of non-rotating and rotating structures. The collection of images for the purpose of creating photogrammetric models can be called more properly, polyoscopy, after Pierre Seguin

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Integration

### Integration  
Photogrammetric data can be complemented with range data from other techniques.  Photogrammetry is more accurate in the x and y direction while range data are generally more accurate in the z direction .  This range data can be supplied by techniques like [LiDAR], laser scanners (using [time of flight], triangulation or [interferometry]), [Structured-light 3D scanner|white-light digitizer]s and any other technique that scans an area and returns x, y, z coordinates for multiple discrete points (commonly called "[point clouds]").  Photos can clearly define the edges of buildings when the point cloud footprint can not.  It is beneficial to incorporate the advantages of both systems and integrate them to create a better product.  
A 3D visualization can be created by georeferencing the aerial photos and LiDAR data in the same reference frame, [image rectification|orthorectifying] the aerial photos, and then draping the orthorectified images on top of the LiDAR grid. It is also possible to create digital terrain models and thus 3D visualisations using pairs (or multiples) of aerial photographs or satellite (e.g. [SPOT satellite] imagery). Techniques such as adaptive least squares stereo matching are then used to produce a dense array of correspondences which are transformed through a camera model to produce a dense array of x, y, z data which can be used to produce [digital terrain model] and [Orthophoto|orthoimage] products. Systems which use these techniques, e.g. the ITG system, were developed in the 1980s and 1990s but have since been supplanted by LiDAR and radar-based approaches, although these techniques may still be useful in deriving elevation models from old aerial photographs or satellite images.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Applications

### Applications
Horatio Nelson bust in [Monmouth Museum], produced using photogrammetry]]
Gibraltar 1 [Neanderthal] skull 3D wireframe model, created with 123d Catch
Photogrammetry is used in fields such as [topographic map]ping, [architecture], [filmmaking], [engineering], [manufacturing], [quality control], [police] investigation, [cultural heritage], and [geology]. [aerial archaeology|Archaeologists] use it to quickly produce plans of large or complex sites, and [meteorologist]s use it to determine the wind speed of [tornado]es when objective weather data cannot be obtained.  
Photograph of person using a controller to explore a 3D photogrammetry experience, Future Cities by DERIVE, recreating Tokyo  
It is also used to combine [live action] with [computer-generated imagery] in movies [post-production]; [The Matrix] is a good example of the use of photogrammetry in film (details are given in the DVD extras). Photogrammetry was used extensively to create photorealistic environmental assets for video games including [The Vanishing of Ethan Carter] as well as [EA DICE]'s [Star Wars Battlefront (2015 video game)|Star Wars Battlefront]. The main character of the game [Hellblade: Senua's Sacrifice] was derived from photogrammetric motion-capture models taken of actress Melina Juergens.  
Photogrammetry is also commonly employed in collision engineering, especially with automobiles. When litigation for a collision occurs and engineers need to determine the exact deformation present in the vehicle, it is common for several years to have passed and the only evidence that remains is crash scene photographs taken by the police. Photogrammetry is used to determine how much the car in question was deformed, which relates to the amount of energy required to produce that deformation. The energy can then be used to determine important information about the crash (such as the velocity at time of impact).  
This technology is also used for the inspection of submerged objects and infrastructure, thanks to its ability to produce precise, georeferenced 3D models. Subsea photogrammetry applications are numerous in the marine energy sector, including integrity monitoring of offshore wind turbine foundations, assessment of anchor chain wear, inspection of cable protection and burial systems, and overall evaluation of offshore structures.It also plays an increasing role in maritime and river civil engineering, particularly for inspecting port quays, dikes, dams, and other hydraulic structures. By providing reliable and reproducible metric data, underwater photogrammetry improves the monitoring of structural changes, optimizes maintenance operations, and reduces the costs associated with interventions in underwater environments.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Applications | Mapping

#### Mapping  
Photomapping is the process of making a map with "cartographic enhancements" that have been drawn from a [Orthophotomosaic|photomosaic] that is "a composite photographic image of the ground," or more precisely, as a controlled photomosaic where "individual photographs are rectified for tilt and brought to a common scale (at least at certain control points)."  
Rectification of imagery is generally achieved by "fitting the projected images of each photograph to a set of four control points whose positions have been derived from an existing map or from ground measurements. When these rectified, scaled photographs are positioned on a grid of control points, a good correspondence can be achieved between them through skillful trimming and fitting and the use of the areas around the principal point where the relief displacements (which cannot be removed) are at a minimum."  
"It is quite reasonable to conclude that some form of photomap will become the standard general map of the future." They go on to suggest that, "photomapping would appear to be the only way to take reasonable advantage" of future data sources like high altitude aircraft and satellite imagery.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Applications | Archaeology

#### Archaeology
Using a pentop computer to photomap an archaeological excavation in the field
Demonstrating the link between [orthophotomap]ping and [archaeology], historic [aerial photography|airphotos] photos were used to aid in developing a reconstruction of the Ventura mission that guided excavations of the structure's walls.  
Photogrammetry has been as used in digital [art restoration], particularly for the virtual reconstruction of historical sculptures and monuments destroyed during [iconiclasm|iconoclastic movements].  
Pteryx UAV, a civilian UAV for aerial photography and photomapping with roll-stabilised camera head  
Overhead photography has been widely applied for mapping surface remains and excavation exposures at archaeological sites. Suggested platforms for capturing these photographs has included: War Balloons from World War I; rubber meteorological balloons; [kite aerial photography|kites]; wooden platforms, metal frameworks, constructed over an excavation exposure; ladders both alone and held together with poles or planks; three legged ladders; single and multi-section poles; bipods; tripods; tetrapods, and aerial bucket trucks ("cherry pickers").  
Handheld, near-nadir, overhead digital photographs have been used with geographic information systems ([Geographic information system|GIS]) to record excavation exposures.  
Photogrammetry is increasingly being used in [maritime archaeology] because of the relative ease of mapping sites compared to traditional methods, allowing the creation of 3D maps which can be rendered in [virtual reality].  
A recent study applied high-resolution photogrammetry in combination with 3D scanning and quantitative morphometric analysis to compare Graeco-Roman funerary masks, enabling the re-identification of fragments and the attribution of archaeological provenience within museum collections.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Applications | 3D modeling

#### 3D modeling
A somewhat similar application is the scanning of objects to automatically make 3D models of them. Since photogrammetry relies on images, there are physical limitations when those images are of an object that has dark, shiny or clear surfaces. In those cases, the produced model often still contains gaps, so additional cleanup with software like [MeshLab], netfabb or MeshMixer is often still necessary. Alternatively, spray painting such objects with matte finish can remove any transparent or shiny qualities.  
[Google Earth] uses photogrammetry to create 3D imagery.  
There is also a project called [Rekrei] that uses photogrammetry to make 3D models of lost/stolen/broken artifacts that are then posted online.  
On [Mount Stanley], an exhibition team sent out by [Project Pressure] created the first ever 3D model of the glacier using drone photography and [Satellite navigation|GNSS] technology showing a surface area decline of 29.5% between 2020 and 2024.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Applications | Rock mechanics

#### Rock mechanics
High-resolution 3D point clouds derived from UAV or ground-based photogrammetry can be used to automatically or semi-automatically extract rock mass properties such as discontinuity orientations, persistence, and spacing.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Photogrammetry | Software

### Software  
There exist many [Software suite|software package]s for photogrammetry; see [comparison of photogrammetry software].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Overview

## Multispectral imaging  
### Overview  
Multispectral imaging captures image data within specific wavelength ranges across the electromagnetic spectrum. The wavelengths may be separated by filters or detected with the use of instruments that are sensitive to particular wavelengths, including light from frequencies beyond the visible light range (i.e. infrared and ultraviolet). It can allow extraction of additional information the human eye fails to capture with its visible receptors for red, green and blue. It was originally developed for military target identification and reconnaissance. Early space-based imaging platforms incorporated multispectral imaging technology to map details of the Earth related to coastal boundaries, vegetation, and landforms. Multispectral imaging has also found use in document and painting analysis.
Multispectral imaging measures light in a small number (typically 3 to 15) of spectral bands.  Hyperspectral imaging is a special case of spectral imaging where often hundreds of contiguous spectral bands are available.  
SDO simultaneously showing sections of the Sun at various wavelengths.]]
Multispectral image of part of the Mississippi River obtained by combining three images acquired at different nominal wavelengths (800nm/infrared, 645nm/red, and 525nm/green) by [Apollo 9] in 1969.
Bek crater and its ray system on the surface of [Mercury (planet)|Mercury], acquired by [MESSENGER], combining images at wavelengths of 996, 748, 433 nm.  The bright yellow patches in other parts of the image are [Hollows (Mercury)|hollows].]]  
Multispectral imaging captures image data within specific [wavelength] ranges across the [electromagnetic spectrum]. The wavelengths may be separated by [Filter (optics)|filters] or detected with the use of instruments that are sensitive to particular wavelengths, including light from [electromagnetic spectrum|frequencies beyond the visible light range] (i.e. [infrared] and [ultraviolet]). It can allow extraction of additional information the human eye fails to capture with its visible receptors for [Trichromacy|red, green and blue]. It was originally developed for military target identification and reconnaissance. Early space-based imaging platforms incorporated multispectral imaging technology to map details of the [Earth] related to coastal boundaries, vegetation, and landforms. Multispectral imaging has also found use in document and painting analysis.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Spectral band usage

### Spectral band usage
> Further: False-color  
For different purposes, different combinations of spectral bands can be used. They are usually represented with red, green, and blue channels. Mapping of bands to colors depends on the purpose of the image and the personal preferences of the analysts. Thermal infrared is often omitted from consideration due to poor spatial resolution, except for special purposes.
* True-color uses only red, green, and blue channels, mapped to their respective colors. As a plain color photograph, it is good for analyzing man-made objects, and is easy to understand for beginner analysts.
* Green-red-infrared, where the blue channel is replaced with near infrared, is used for vegetation, which is highly reflective in near IR; it then shows as blue. This combination is often used to detect vegetation and camouflage.
* Blue-NIR-MIR, where the blue channel uses visible blue, green uses NIR (so vegetation stays green), and MIR is shown as red. Such images allow the water depth, vegetation coverage, soil moisture content, and the presence of fires to be seen, all in a single image.
Many other combinations are in use. NIR is often shown as red, causing vegetation-covered areas to appear red.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Spectral band usage | Typical spectral bands

#### Typical spectral bands
The wavelengths are approximate; exact values depend on the particular instruments (e.g. characteristics of satellite's sensors for Earth observation, characteristics of illumination and sensors for document analysis):
* Blue, 450–515/520&nbsp;nm, is used for atmosphere and deep water imaging, and can reach depths up to  in clear water.
* Green, 515/520–590/600&nbsp;nm, is used for imaging vegetation and deep water structures, up to  in clear water.
* Red, 600/630–680/690&nbsp;nm, is used for imaging man-made objects, in water up to  deep, soil, and vegetation.
* Near infrared (NIR), 750–900&nbsp;nm, is used primarily for imaging vegetation.
* Mid-infrared (MIR), 1550–1750&nbsp;nm, is used for imaging vegetation, soil moisture content, and some [forest fire]s.
* Far-infrared (FIR), 2080–2350&nbsp;nm, is used for imaging soil, moisture, geological features, silicates, clays, and fires.
* [thermography|Thermal infrared], 10,400–12,500&nbsp;nm, uses emitted instead of reflected radiation to image geological structures, thermal differences in water currents, fires, and for night studies.
* [Radar] and related technologies are useful for mapping terrain and for detecting various objects.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Classification

### Classification
Unlike other [aerial photographic and satellite image interpretation] work, these multispectral images do not make it easy to identify directly the feature type by visual inspection. Hence the remote sensing data has to be classified first, followed by processing by various data enhancement techniques so as to help the user to understand the features that are present in the image.  
Such classification is a complex task which involves rigorous validation of the training samples depending on the classification algorithm used. The techniques can be grouped mainly into two types.
* Supervised classification techniques
* Unsupervised classification techniques  
[Supervised classification] makes use of training samples. Training samples are areas on the ground for which there is [ground truth], that is, what is there is known. The [spectral signature]s of the training areas are used to search for similar signatures in the remaining pixels of the image, and we will classify accordingly.  This use of training samples for classification is called supervised classification.  Expert knowledge is very important in this method since the selection of the training samples and a biased selection can badly affect the accuracy of classification.  Popular techniques include the [maximum likelihood principle] and [convolutional neural network]. The maximum likelihood principle calculates the probability of a pixel belonging to a class (i.e. feature) and allots the [pixel] to its most probable class. Newer [convolutional neural network] based methods  account for both spatial proximity and entire spectra to determine the most likely class.  
In case of [unsupervised classification] no prior knowledge is required for classifying the features of the image. The natural clustering or grouping of the pixel values (i.e. the gray levels of the pixels) are observed. Then a threshold is defined for adopting the number of classes in the image. The finer the threshold value, the more classes there will be. However, beyond a certain limit the same class will be represented in different classes in the sense that variation in the class is represented. After forming the clusters, [ground truth] validation is done to identify the class the image pixel belongs to. Thus in this unsupervised classification a priori information about the classes is not required.  One of the popular methods in unsupervised classification is [k-means clustering].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Classification | Data analysis software

#### Data analysis software
* MicroMSI is endorsed by the [National Geospatial-Intelligence Agency|NGA].
* [Opticks (Software)|Opticks] is an open-source remote sensing application.
* Multispec is freeware multispectral analysis software.
* Gerbil is open source multispectral visualization and analysis software.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Applications | Military target tracking

#### Military target tracking
Multispectral imaging measures light emission and is often used in detecting or tracking military targets. In 2003, researchers at the [United States Army Research Laboratory] and the Federal Laboratory Collaborative Technology Alliance reported a dual band multispectral imaging [Focal-plane array testing|focal plane array] (FPA). This FPA allowed researchers to look at two infrared (IR) planes at the same time.  Because mid-wave infrared (MWIR) and long wave infrared (LWIR) technologies measure radiation inherent to the object and require no external light source, they also are referred to as [Thermography|thermal imaging] methods.  
The brightness of the image produced by a thermal imager depends on the objects [emissivity] and temperature.&nbsp; Every material has an [infrared signature] that aids in the identification of the object. These signatures are less pronounced in [Hyperspectral imaging|hyperspectral] systems (which image in many more bands than multispectral systems) and when exposed to wind and, more dramatically, to rain. Imaging systems that use MWIR technology function better with solar reflections on the target's surface and produce more definitive images of hot objects, such as engines, compared to LWIR technology. However, LWIR operates better in hazy environments like smoke or fog because less [scattering] occurs in the longer wavelengths. Usually, [Earth observation satellite]s have three or more [radiometer]s. Each acquires one digital image (in remote sensing, called a 'scene') in a small spectral band. The bands are grouped into wavelength regions based on the origin of the light and the interests of the researchers.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | Applications | Weather forecasting

#### Weather forecasting
Modern weather satellites produce imagery in a variety of spectra.  
In the case of [Landsat] satellites, several different band designations have been used, with as many as 11 bands ([Landsat 8]) comprising a multispectral image. [Spectral imaging] with a higher radiometric resolution
(involving hundreds or thousands of bands), finer spectral resolution (involving smaller bands), or wider spectral coverage may be called [hyperspectral] or ultraspectral. The painting is irradiated by [ultraviolet], visible and [infrared] rays and the reflected radiation is recorded in a camera sensitive in this region of the spectrum. The image can also be registered using the transmitted instead of reflected radiation. In special cases the painting can be irradiated by [UV rays|UV], VIS or IR rays and the [fluorescence] of [pigments] or [varnishes] can be registered.  
Multispectral analysis has assisted in the interpretation of [Herculaneum papyri|ancient papyri], such as those found at [Herculaneum], by imaging the fragments in the infrared range (1000&nbsp;nm). Often, the text on the documents appears to the naked eye as black ink on black paper. At 1000&nbsp;nm, the difference in how paper and ink reflect infrared light makes the text clearly readable. It has also been used to image the [Archimedes palimpsest] by imaging the parchment leaves in bandwidths from 365–870&nbsp;nm, and then using advanced digital image processing techniques to reveal the undertext with Archimedes' work. Multispectral imaging has been used in a [Andrew W. Mellon Foundation|Mellon Foundation] project at [Yale University] to compare inks in medieval English manuscripts.  
Multispectral imaging has also been used to examine discolorations and stains on old books and manuscripts.  Comparing the "spectral fingerprint" of a stain to the characteristics of known chemical substances can make it possible to identify the stain. This technique has been used to examine medical and [alchemical] texts, seeking hints about the activities of early chemists and the possible chemical substances they may have used in their experiments. Like a cook spilling flour or vinegar on a cookbook, an early chemist might have left tangible evidence on the pages of the ingredients used to make medicines.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Multispectral imaging | See also

### See also
* [Hyperspectral imaging]
* [Imaging spectrometer]
* [Imaging spectroscopy]
* [Liquid crystal tunable filter]
* [Multispectral pattern recognition]
* [Normalized difference vegetation index] (NDVI)
* [Pansharpening]
* [Reconnaissance satellite]
* [Remote sensing]
* [Satellite imagery]

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | National Collection of Aerial Photography | Overview

## National Collection of Aerial Photography  
### Overview  
The National Collection of Aerial Photography is a photographic archive in Edinburgh, Scotland, containing over 30 million aerial photographs of worldwide historic events and places. From 2008–2015 it was part of the Royal Commission on the Ancient and Historical Monuments of Scotland and since then it has been a sub-brand of Historic Environment Scotland.  Many of the aerial reconnaissance photographs were taken during the Second World War and the Cold War, and were declassified and released by the Ministry of Defence. The collection also contains over 1.8 million aerial survey photographs of Scotland, during and in the years after the Second World War, as well as post-war Ordnance Survey, over 4 million photogrammetric images, and over 10 million aerial survey images of international sites as part of The Aerial Reconnaissance Archives (TARA). The collection contains both military declassified and non-military aerial photographs from over a dozen different national and international organisations.
NCAP’s historical aerial photography is primarily used to locate unexploded Second World War bombs by European bomb disposal companies and in historical, archaeological and climate change research. It is also used for documentaries and dramas on television and in film.  
NCAP logo
The National Collection of Aerial Photography is a photographic [archive] in Edinburgh, Scotland, containing over 30 million aerial photographs of worldwide historic events and places. From 2008–2015 it was part of the [Royal Commission on the Ancient and Historical Monuments of Scotland] and since then it has been a sub-brand of [Historic Environment Scotland].  Many of the [aerial reconnaissance] photographs were taken during the [World War II|Second World War] and the [Cold War], and were declassified and released by the [Ministry of Defence (United Kingdom)|Ministry of Defence]. The collection also contains over 1.8 million [aerial survey] photographs of Scotland, during and in the years after the Second World War, as well as post-war [Ordnance Survey], over 4 million [photogrammetry|photogrammetric] images, and over 10 million aerial survey images of international sites as part of The Aerial Reconnaissance Archives (TARA). The collection contains both military [declassification|declassified] and non-military aerial photographs from over a dozen different national and international organisations.  
NCAP’s historical aerial photography is primarily used to locate unexploded Second World War bombs by European bomb disposal companies and in historical, archaeological and [climate change] research. It is also used for documentaries and dramas on television and in film.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | National Collection of Aerial Photography | Collections

### Collections
*[Airbus|AIRBUS] Defence & Space
*Allied Central Interpretation Unit (ACIU) (from [RAF Medmenham])
*[Defence Geographic Centre]
*[Directorate of Overseas Surveys|Directorate of Overseas Survey]s (DOS)
*[Environment Agency]
*[German Air Force]
*[Digimap|Getmapping]
*[Defence Intelligence Fusion Centre|Joint Air Reconnaissance Intelligence Centre] (JARIC)
*[Northwest African Photographic Reconnaissance Wing|Mediterranean Allied Photo Reconnaissance Wing] (MAPRW)
*[National Archives and Records Administration]
*[Natural Environment Research Council]
*Scottish Office Air Photographs Unit

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | National Collection of Aerial Photography | Collections | Bibliography

#### Bibliography
*Cowley, Dave C, Crawford, James (2009). Above Scotland: The National Collection of Aerial Photography. [Royal Commission on the Ancient and Historical Monuments of Scotland|RCAHMS].
*Bailey, Rebecca M, Crawford, James, Williams, Allan (2010). Above Scotland Cities: The National Collection of Aerial Photography. [Royal Commission on the Ancient and Historical Monuments of Scotland|RCAHMS].
*Crawford, James (2012). Scotland's Landscapes: The National Collection of Aerial Photography. [Royal Commission on the Ancient and Historical Monuments of Scotland|RCAHMS].
*Hanson, William S., Oltean, Ioana A. editors (2012). Archaeology from Historical Aerial and Satellite Archives, [Springer Science+Business Media|Springer Science & Business Media].
*Williams, Allan (2013). Operation Crossbow: The Untold Story of Photographic Intelligence and the Search For Hitler's V Weapons. [Random House].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial image library | Overview

## Aerial image library  
### Overview  
An aerial image library is a collection of aerial imagery. The imagery is taken from cameras placed on aircraft, which capture images of the structures and features of the land below. These libraries can contain millions of individual images which depict geographic areas in incredible detail.
Aerial image libraries can provide a wealth of information to users of the imagery. Governments often use such libraries to maintain current records of construction and in conducting property assessment. Insurance companies can also use aerial image libraries to maintain records of natural disaster-related damages. Utilities companies may also keep libraries of electric corridors and pipeline networks to plan expansion or maintenance.
Using computer software, aerial image libraries can stitch together images to create lifelike maps of geographic regions. These maps can be either orthogonal (top-down images) or oblique (images captured at an angle). If these images are geo-referenced, users can determine precisely where the structures and features depicted are located on the earth.  
An aerial image library is a collection of [aerial imagery]. The imagery is taken from cameras placed on aircraft, which capture images of the structures and features of the land below. These libraries can contain millions of individual images which depict geographic areas in incredible detail.  
Aerial image libraries can provide a wealth of information to users of the imagery. Governments often use such libraries to maintain current records of [construction] and in conducting [tax assessment|property assessment]. [Insurance] companies can also use aerial image libraries to maintain records of [natural disaster]-related damages. [Utilities] companies may also keep libraries of [electric power transmission|electric corridors] and [Pipeline transport|pipeline] networks to plan expansion or maintenance.  
Using computer software, aerial image libraries can stitch together images to create lifelike maps of geographic regions. These maps can be either [orthogonal] (top-down images) or [:wikt:oblique|oblique] (images captured at an angle). If these images are [georeference|geo-referenced], users can determine precisely where the structures and features depicted are located on the earth.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Timeline of first Earth observation satellites | Overview

## Timeline of first Earth observation satellites  
### Overview  
The timeline of first Earth observation satellites shows, in chronological order, those successful Earth observation satellites, that is, Earth satellites with a program of Earth science. Sputnik 1, while the first satellite ever launched, did not conduct Earth science. Explorer 1 was the first satellite to make an Earth science discovery when it found the Van Allen belts.  
The timeline of first Earth observation satellites shows, in chronological order, those successful [Earth observation satellite]s, that is, [Earth satellite]s with a program of [Earth science]. [Sputnik 1], while the first satellite ever launched, did not conduct Earth science. [Explorer 1] was the first satellite to make an Earth science discovery when it found the [Van Allen belts].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Timeline of first Earth observation satellites | 1950s

### 1950s  
{| class="wikitable sortable sticky-header"  
! Satellite
! Country
! Date
! Organization
! Notes

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Timeline of first Earth observation satellites | 1960s

### 1960s  
{| class="wikitable sortable sticky-header"
! Satellite
! Country
! Date
! Organization
! Notes

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Overview

## Image moment  
### Overview  
In image processing, computer vision and related fields, an image moment is a certain particular weighted average (moment) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation.
Image moments are useful to describe objects after segmentation. Simple properties of the image which are found via image moments include area (or total intensity), its centroid, and information about its orientation.  
In [image processing], [computer vision] and related fields, an image moment is a certain particular [Weighted arithmetic mean|weighted average] ([moment (mathematics)|moment]) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation.  
Image moments are useful to describe objects after [Image segmentation|segmentation]. [#Examples|Simple properties of the image] which are found via image moments include area (or total intensity), its [centroid], and [#Examples 2|information about its orientation].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Raw moments

### Raw moments
For a 2D continuous function f(x,y) the [Moment (mathematics)|moment] (sometimes called "raw moment") of order (p + q) is defined as  
:  M_{pq}=\int\limits_{-\infty}^{\infty} \int\limits_{-\infty}^{\infty} x^py^qf(x,y) \,dx\, dy  
for p,q = 0,1,2,...
Adapting this to scalar ([grayscale]) image with pixel intensities I(x,y), raw image moments Mij are calculated by  
: M_{ij} = \sum_x \sum_y x^i y^j I(x,y)\,\!  
In some cases, this may be calculated by considering the image as a [probability density function], i.e., by dividing the above by  
: \sum_x \sum_y I(x,y) \,\!  
A uniqueness theorem states that if f(x,y)
is piecewise continuous and has nonzero values only in a finite part of the xy
plane, moments of all orders exist, and the moment sequence (Mpq) is uniquely determined by f(x,y). Conversely, (Mpq) uniquely determines f(x,y).  In practice, the image is summarized with functions of a few lower order moments.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Raw moments | Examples

#### Examples  
Simple image properties derived via raw moments include:
* Area (for binary images) or sum of grey level (for greytone images):   M_{00}
* Centroid:   \{\bar{x},\ \bar{y} \} = \left\{ \frac{M_{10}}{M_{00}}, \frac{M_{01}}{ M_{00}} \right\}

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Central moments

### Central moments
[moment about the mean|Central moments] are defined as  
:  \mu_{pq} = \int\limits_{-\infty}^{\infty} \int\limits_{-\infty}^{\infty} (x - \bar{x})^p(y - \bar{y})^q f(x,y) \, dx \, dy  
where  \bar{x}=\frac{M_{10}}{M_{00}}  and  \bar{y}=\frac{M_{01}}{M_{00}}  are the components of the [centroid].  
If &fnof;(x,&nbsp;y) is a digital image, then the previous equation becomes  
: \mu_{pq} = \sum_{x} \sum_{y} (x - \bar{x})^p(y - \bar{y})^q f(x,y)  
The central moments of order up to 3 are:  
\begin{align}
\mu_{00} &= M_{00}, &
\mu_{01} &= 0, \\
\mu_{10} &= 0, &
\mu_{11} &= M_{11} - \bar{x} M_{01} = M_{11} - \bar{y} M_{10}, \\
\mu_{20} &= M_{20} -  \bar{x} M_{10},  &
\mu_{02} &= M_{02} -  \bar{y} M_{01},  \\
\mu_{21} &= M_{21} - 2 \bar{x} M_{11} - \bar{y} M_{20} + 2 \bar{x}^2 M_{01},  &
\mu_{12} &= M_{12} - 2 \bar{y} M_{11} - \bar{x} M_{02} + 2 \bar{y}^2 M_{10},  \\
\mu_{30} &= M_{30} - 3 \bar{x} M_{20} + 2 \bar{x}^2 M_{10},  &
\mu_{03} &= M_{03} - 3 \bar{y} M_{02} + 2 \bar{y}^2 M_{01}.
\end{align}  
It can be shown that:
: \mu_{pq} = \sum_{m}^p \sum_{n}^q {p\choose m} {q\choose n}(-\bar{x})^{(p-m)}(-\bar{y})^{(q-n)}  M_{mn}  
Central moments are [Translational invariance|translational invariant].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Central moments | Examples

#### Examples  
Information about image orientation can be derived by first using the second order central moments to construct a [covariance matrix].  
\begin{align}
\mu'_{20} &= \mu_{20} / \mu_{00} = M_{20}/M_{00} - \bar{x}^2 \\
\mu'_{02} &= \mu_{02} / \mu_{00} = M_{02}/M_{00} - \bar{y}^2 \\
\mu'_{11} &= \mu_{11} / \mu_{00} = M_{11}/M_{00} - \bar{x}\bar{y}
\end{align}  
The [covariance matrix] of the image  I(x,y)  is now  
: \operatorname{cov}[I(x,y)] = \begin{bmatrix} \mu'_{20}  & \mu'_{11} \\ \mu'_{11} & \mu'_{02} \end{bmatrix}.  
The [eigenvector]s of this matrix correspond to the major and minor axes of the image intensity, so the orientation can thus be extracted from the angle of the eigenvector associated with the largest eigenvalue towards the axis closest to this eigenvector. It can be shown that this angle Θ is given by the following formula:  
: \Theta = \frac{1}{2} \arctan \left( \frac{2\mu'_{11}}{\mu'_{20} - \mu'_{02}} \right)  
The above formula holds as long as:
: \mu'_{20} - \mu'_{02} \ne 0  
The [eigenvalue]s of the covariance matrix can easily be shown to be  
:  \lambda_i = \frac{\mu'_{20} + \mu'_{02}}{2}  \pm \frac{\sqrt{4{\mu'}_{11}^2 + ({\mu'}_{20}-{\mu'}_{02})^2  }}{2},  
and are proportional to the squared length of the eigenvector axes.  The relative difference in magnitude of the eigenvalues are thus an indication of the eccentricity of the image, or how elongated it is. The [Eccentricity (mathematics)|eccentricity] is  
:  \sqrt{1 - \frac{\lambda_2}{\lambda_1}}.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Moment invariants

### Moment invariants  
Moments are well-known for their application in image analysis, since they can be used to derive [Invariant (mathematics)|invariants] with respect to specific transformation classes.  
The term invariant moments is often abused in this context. However, while moment invariants are invariants that are formed from moments, the only moments that are invariants themselves are the central moments.  
Note that the invariants detailed below are exactly invariant only in the continuous domain. In a discrete domain, neither scaling nor rotation are well defined: a discrete image transformed in such a way is generally an approximation, and the transformation is not reversible. These invariants therefore are only approximately invariant when describing a shape in a discrete image.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Moment invariants | Translation invariants

#### Translation invariants
The central moments &mu;i j of any order are, by construction, invariant with respect to [translation (geometry)|translations].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Moment invariants | Scale invariants

#### Scale invariants  
Invariants &eta;i j with respect to both [translation (geometry)|translation] and [Scale (ratio)|scale] can be constructed from central moments by dividing through a properly scaled zero-th central moment:  
: \eta_{ij} = \frac{\mu_{ij}}
{\mu_{00}^{\left(1 + \frac{i+j}{2}\right)}}\,\!  
where i + j ≥ 2.
Note that translational invariance directly follows by only using central moments.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Moment invariants | Rotation invariants

#### Rotation invariants  
As shown in the work of Hu,
invariants with respect to [translation (geometry)|translation], [Scale (ratio)|scale], and [rotation] can be constructed:  
I_1 = \eta_{20} + \eta_{02}  
I_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2  
I_3 = (\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2  
I_4 = (\eta_{30} + \eta_{12})^2 + (\eta_{21} + \eta_{03})^2  
I_5 = (\eta_{30} - 3\eta_{12}) (\eta_{30} + \eta_{12})[ (\eta_{30} + \eta_{12})^2 - 3 (\eta_{21} + \eta_{03})^2] + (3 \eta_{21} - \eta_{03}) (\eta_{21} + \eta_{03})[ 3(\eta_{30} + \eta_{12})^2 -  (\eta_{21} + \eta_{03})^2]  
I_6 =  (\eta_{20} - \eta_{02})[(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2] + 4\eta_{11}(\eta_{30} + \eta_{12})(\eta_{21} + \eta_{03})  
I_7 = (3 \eta_{21} - \eta_{03})(\eta_{30} + \eta_{12})[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2] - (\eta_{30} - 3\eta_{12})(\eta_{21} + \eta_{03})[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2].  
These are well-known as Hu moment invariants.  
The first one, I1, is analogous to the [moment of inertia] around the image's centroid, where the pixels' intensities are analogous to physical density. The first six,  I1 ... I6, are reflection symmetric, i.e. they are unchanged if the image is changed to a mirror image. The last one, I7, is reflection antisymmetric (changes sign under reflection), which enables it to distinguish mirror images of otherwise identical images.  
A general theory on deriving complete and independent sets of rotation moment invariants was proposed by J. Flusser. He showed that the traditional set of Hu moment invariants is neither independent nor complete. I3 is not very useful as it is dependent on the others ( I_3 = (I_5^2 + I_7^2) / I_4^3 ). In the original Hu's set there is a missing third order independent moment invariant:
:
I_8 = \eta_{11}[ ( \eta_{30} + \eta_{12})^2 - (\eta_{03} + \eta_{21})^2  ] - (\eta_{20}-\eta_{02}) (\eta_{30}+\eta_{12}) (\eta_{03}+\eta_{21})  
Like I7, I8 is also reflection antisymmetric.  
Later, J. Flusser and T. Suk specialized the theory for N-rotationally symmetric shapes case.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Image moment | Applications

### Applications  
Zhang et al. applied Hu moment invariants to solve the [pathological brain detection] (PBD) problem.
Doerr and Florence used information of the object orientation related to the second order central moments to effectively extract translation- and rotation-invariant object cross-sections from micro-X-ray tomography image data.  
D. A. Hoeltzel and Wei-Hua Chieng used Hu moment invariant to perform on a dimensionally-parameterized four bar mechanism which yielded 15 distinct coupler curve groups (patterns) from a total of 356 generated coupler curves.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | Overview

## Earth observation satellite  
### Overview  
An Earth observation satellite or Earth remote sensing satellite is a satellite used or designed for Earth observation (EO) from orbit, including spy satellites and similar ones intended for non-military uses such as environmental monitoring, meteorology, cartography and others. The most common type are Earth imaging satellites, that take satellite images, analogous to aerial photographs; some EO satellites may perform remote sensing without forming pictures, such as in GNSS radio occultation.
The first occurrence of satellite remote sensing can be dated to the launch of the first artificial satellite, Sputnik 1, by the Soviet Union on October 4, 1957. Sputnik 1 sent back radio signals, which scientists used to study the ionosphere.
The United States Army Ballistic Missile Agency launched the first American satellite, Explorer 1, for NASA's Jet Propulsion Laboratory on January 31, 1958. The information sent back from its radiation detector led to the discovery of the Earth's Van Allen radiation belts. The TIROS-1 spacecraft, launched on April 1, 1960, as part of NASA's Television Infrared Observation Satellite (TIROS) program, sent back the first television footage of weather patterns to be taken from space.
In 2008, more than 150 Earth observation satellites were in orbit, recording data with both passive and active sensors and acquiring more than 10 terabits of data daily. By 2021, that total had grown to over 950, with the largest number of satellites operated by US-based company Planet Labs.
Most Earth observation satellites carry instruments that should be operated at a relatively low altitude. Most orbit at altitudes above 500 to 600 kilometers (310 to 370 mi). Lower orbits have significant air-drag, which makes frequent orbit reboost maneuvers necessary. The Earth observation satellites ERS-1, ERS-2 and Envisat of European Space Agency as well as the MetOp spacecraft of EUMETSAT are all operated at altitudes of about 800 km (500 mi). The Proba-1, Proba-2 and SMOS spacecraft of European Space Agency are observing the Earth from an altitude of about 700 km (430 mi). The Earth observation satellites of UAE, DubaiSat-1 & DubaiSat-2 are also placed in Low Earth orbits (LEO) orbits and providing satellite imagery of various parts of the Earth.
To get global coverage with a low orbit, a polar orbit is used. A low orbit will have an orbital period of about 100 minutes and the Earth will rotate around its polar axis about 25° between successive orbits. The ground track moves towards the west 25° each orbit, allowing a different section of the globe to be scanned with each orbit. Most are in Sun-synchronous orbits.
A geostationary orbit, at 36,000 km (22,000 mi), allows a satellite to hover over a constant spot on the earth since the orbital period at this altitude is 24 hours. This allows uninterrupted coverage of more than 1/3 of the Earth per satellite, so three satellites, spaced 120° apart, can cover the whole Earth. This type of orbit is mainly used for meteorological satellites.  
A-train satellite constellation as of 2014]]  
An Earth observation satellite or Earth remote sensing satellite is a [satellite] used or designed for [Earth observation] (EO) from [orbit], including [spy satellite]s and similar ones intended for non-military uses such as [environmental monitoring], [meteorology], [cartography] and others. The most common type are Earth imaging satellites, that take [satellite image]s, analogous to [aerial photograph]s; some EO satellites may perform [remote sensing] without forming pictures, such as in [GNSS radio occultation].  
The first occurrence of satellite remote sensing can be dated to the launch of the first artificial satellite, [Sputnik 1], by the Soviet Union on October 4, 1957.  
The United States Army Ballistic Missile Agency launched the first American satellite, [Explorer 1], for NASA's Jet Propulsion Laboratory on January 31, 1958. The information sent back from its radiation detector led to the discovery of the Earth's [Van Allen radiation belt]s. The [TIROS-1] spacecraft, launched on April 1, 1960, as part of NASA's [Television Infrared Observation Satellite] (TIROS) program, sent back the first television footage of weather patterns to be taken from space. By 2021, that total had grown to over 950, with the largest number of satellites operated by US-based company [Planet Labs].  
Most [Earth] observation satellites carry instruments that should be operated at a relatively low altitude. Most orbit at altitudes above . Lower orbits have significant [Drag (physics)|air-drag], which makes frequent orbit [reboost] maneuvers necessary. The Earth observation satellites [European Remote-Sensing Satellite|ERS-1, ERS-2] and [Envisat] of [European Space Agency] as well as the [MetOp] spacecraft of [EUMETSAT] are all operated at altitudes of about . The [PROBA|Proba-1], [Proba-2] and [Soil Moisture and Ocean Salinity satellite|SMOS] spacecraft of European Space Agency are observing the Earth from an altitude of about . The Earth observation satellites of UAE, [DubaiSat-1] & [DubaiSat-2] are also placed in [Low Earth orbit]s (LEO) orbits and providing [satellite imagery] of various parts of the Earth.  
To get global coverage with a low orbit, a [polar orbit] is used. A low orbit will have an orbital period of about 100 minutes and the Earth will rotate around its polar axis about 25° between successive orbits. The [ground track] moves towards the west 25° each orbit, allowing a different section of the globe to be scanned with each orbit. Most are in [Sun-synchronous orbit]s.  
A [geostationary orbit], at , allows a satellite to hover over a constant spot on the earth since the orbital period at this altitude is 24 hours. This allows uninterrupted coverage of more than 1/3 of the Earth per satellite, so three satellites, spaced 120° apart, can cover the whole Earth. This type of orbit is mainly used for [Weather satellite|meteorological satellites].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | History

### History
> See also: Remote sensing#History  
CORONA 98, 1965]]
[Herman Potočnik] explored the idea of using orbiting spacecraft for detailed peaceful and military observation of the ground in his 1928 book, The Problem of Space Travel. He described how the special conditions of space could be useful for scientific experiments. The book described [geostationary] satellites (first put forward by [Konstantin Tsiolkovsky]) and discussed communication between them and the ground using radio, but fell short of the idea of using satellites for mass broadcasting and as telecommunications relays.  
The onset of the [Cold War] prompted the rapid development of [Launch vehicle|Satellite launch systems] and camera technology capable of sufficient Earth observation to garner intelligence on enemy military infrastructure and evaluate nuclear posture. Following the U-2 incident in 1960, which highlighted the risks of aerial spying, the U.S. accelerated surveillance satellite programs like [CORONA (satellite)|CORONA]. Satellites largely replaced aircraft overflights for surveillance after 1960.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | Applications | Weather

### Applications  
#### Weather
> Main: Weather satellite  
> See also: Satellite temperature measurements  
GOES-8, a [United States] weather satellite  
A weather satellite is a type of [satellite] that is primarily used to monitor the [weather] and [climate] of the [Earth]. These meteorological satellites, however, see more than [cloud]s and cloud systems. City lights, [fire]s, effects of [pollution], [auroral light|aurora]s, [Dust storm|sand and dust storms], [snow] cover, [ice] mapping, boundaries of [ocean current]s, [energy] flows, etc., are other types of environmental information collected using weather satellites.  
Weather satellite images helped in monitoring the volcanic ash cloud from [Mount St. Helens] and activity from other volcanoes such as [Mount Etna]. Smoke from fires in the western United States such as [Colorado] and [Utah] have also been monitored.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | Applications | Environmental monitoring

#### Environmental monitoring
Composite satellite image of the Earth, showing its entire surface in equirectangular projection  
Other environmental satellites can assist [environmental monitoring] by detecting changes in the Earth's vegetation, atmospheric trace gas content, sea state, ocean color, and ice fields. By monitoring vegetation changes over time, droughts can be monitored by comparing the current vegetation state to its long term average. For example, the 2002 oil spill off the northwest coast of [Spain] was watched carefully by the European [Envisat|ENVISAT], which, though not a weather satellite, flies an instrument (ASAR) which can see changes in the sea surface. Anthropogenic emissions can be monitored by evaluating data of tropospheric NO2 and SO2.  
These types of satellites are almost always in [Sun-synchronous orbit|Sun-synchronous] and [Frozen orbit|"frozen"] orbits. A Sun-synchronous orbit passes over each spot on the ground at the same time of day, so that observations from each pass can be more easily compared, since the Sun is in the same spot in each observation. A [Frozen orbit|"frozen"] orbit is the closest possible orbit to a circular orbit that is undisturbed by the [Geopotential model|oblateness of the Earth], gravitational attraction from the Sun and Moon, [solar radiation pressure], and [air drag].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | Applications | Mapping

#### Mapping
Terrain can be mapped from space with the use of satellites, such as [Radarsat-1] and [TerraSAR-X].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | International regulations

### International regulations
RapidEye Earth exploration-satellite system in action around the Earth  
According to the [International Telecommunication Union] (ITU), Earth exploration-satellite service (also: Earth exploration-satellite radiocommunication service) is – according to Article 1.51 of the [ITU Radio Regulations] (RR) – defined as:
A [radiocommunication service] between [earth station]s and one or more [radio space station|space station]s, which may include links between space stations, in which:
*information relating to the characteristics of the Earth and its natural phenomena, including data relating to the state of the environment, is obtained from passive or [Radar|active sensors] on [satellites];
*similar information is collected from airborne or Earth-based platforms;
*such information may be distributed to earth stations within the system concerned;
*platform interrogation may be included.
This service may also include feeder links necessary for its operation.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | International regulations | Classification

#### Classification
This radiocommunication service is classified in accordance with ITU Radio Regulations (article 1) as follows: <br />
[Fixed service] (article 1.20)
*[Fixed-satellite service] (article 1.21)
*[Inter-satellite service] (article 1.22)
*Earth exploration-satellite service
**[Meteorological-satellite service] (article 1.52)

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | International regulations | Frequency allocation

#### Frequency allocation
The allocation of radio frequencies is provided according to Article 5 of the ITU Radio Regulations (edition 2012).  
In order to improve harmonisation in spectrum utilisation, the majority of service-allocations stipulated in this document were incorporated in national Tables of Frequency Allocations and Utilisations which is with-in the responsibility of the appropriate national administration. The allocation might be primary, secondary, exclusive, and shared.
*primary allocation:  is indicated by writing in capital letters (see example below)
*secondary allocation: is indicated by small letters
*exclusive or shared utilization: is within the responsibility of administrations
However, military usage, in bands where there is civil usage, will be in accordance with the ITU Radio Regulations.  
; Example of [frequency allocation]:
{| class=wikitable  
::::: SPACE OPERATION (space-to-Earth) <br />EARTH EXPLORATION-SATELLITE (Earth-to-space) <br />METEOROLOGICAL-SATELLITE (Earth-to-space)<br />Fixed <br />Mobile except aeronautical mobile  
::::: RADIOLOCATION<br />SPACE RESEARCH<br />Standard frequency and time signal-satellite (Earth-to-space)

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Earth observation satellite | See also

### See also  
* [Committee on Earth Observation Satellites]
* [Data collection satellite]
* [Earth observation]
* [Earth observation satellites transmission frequencies]
* [Earth Observing System] - a NASA program comprising a series of satellite missions
* [First images of Earth from space]
* [Satellite imagery#Imaging satellites|Imaging satellites]
* [List of Earth observation satellites]
* [Space telescope]
* [Satellite imagery]
*[GNSS radio occultation]
*[Microwave radiometer#Spaceborne]
*[Radar earth observation satellite]
**[Radar imaging]
**[Synthetic-aperture radar]
***[Interferometric synthetic-aperture radar]
*[Satellite altimetry]

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Overview

## Remote sensing (oceanography)  
### Overview  
Remote sensing in oceanography is a widely used observational technique which enables researchers to acquire data of a location without physically measuring at that location. Remote sensing in oceanography mostly refers to measuring properties of the ocean surface with sensors on satellites or planes, which compose an image of captured electromagnetic radiation. A remote sensing instrument can either receive radiation from the Earth's surface (passive), whether reflected from the Sun or emitted, or send out radiation to the surface and catch the reflection (active). All remote sensing instruments carry a sensor to capture the intensity of the radiation at specific wavelength windows, to retrieve a spectral signature for every location. The physical and chemical state of the surface determines the emissivity and reflectance for all bands in the electromagnetic spectrum, linking the measurements to physical properties of the surface. Unlike passive instruments, active remote sensing instruments also measure the two-way travel time of the signal; which is used to calculate the distance between the sensor and the imaged surface. Remote sensing satellites often carry other instruments which keep track of their location and measure atmospheric conditions.
Remote sensing observations, in comparison to (most) physical observations, are consistent in time and have good spatial coverage. Since the ocean is fluid, it is constantly changing on different spatial and temporal scales. Capturing the spatial variation of the ocean with remote sensing is considered extremely valuable and is on the frontier of oceanographic research. The high variability of the ocean surface is also the deterministic factor in the differences between land and ocean remote sensing.  
> Main: Remote sensing  
Remote sensing in oceanography is a widely used [Observational techniques|observational technique] which enables researchers to acquire data of a location without physically measuring at that location. Remote sensing in oceanography mostly refers to measuring properties of the [ocean] surface with sensors on [satellite]s or planes, which compose an image of captured [electromagnetic radiation]. A remote sensing instrument can either receive radiation from the Earth's surface (passive), whether reflected from the Sun or emitted, or send out radiation to the surface and catch the reflection (active). All remote sensing instruments carry a sensor to capture the intensity of the radiation at specific [wavelength] windows, to retrieve a [spectral signature] for every location. The physical and chemical state of the surface determines the [emissivity] and [reflectance] for all bands in the [electromagnetic spectrum], linking the measurements to physical properties of the surface. Unlike passive instruments, active remote sensing instruments also measure the [two-way travel time] of the signal; which is used to calculate the distance between the sensor and the imaged surface. Remote sensing satellites often carry other instruments which keep track of their location and measure atmospheric conditions.  
Remote sensing observations, in comparison to (most) physical observations, are consistent in time and have good spatial coverage. Since the ocean is fluid, it is constantly changing on different spatial and temporal scales. Capturing the spatial variation of the ocean with remote sensing is considered extremely valuable and is on the frontier of oceanographic research. The high variability of the ocean surface is also the deterministic factor in the differences between land and ocean remote sensing.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing of the ocean | Characteristics

### Remote sensing of the ocean  
#### Characteristics
Remote sensing is actively used in various fields of natural sciences like [Remote sensing (geology)|geology], physical geography, ecology, [Remote sensing (archaeology)|archeology] and meteorology but, remote sensing of the ocean is vastly different. (except for visible light) therefore the ocean surface is easy to monitor but it is a challenge to retrieve information of deeper layers. Remote sensing enables temporal analysis over vast spatial scale, since satellites have a constant [revisit time], provide a wide image and are often operational for multiple consecutive years. This concept of constant data in time and space was a breakthrough in [oceanography], which previously relied on measurements from [Drifter (floating device)|drifters], coastal locations like [tide gauge]s, ships and [buoy]s. All in-situ measurements either have a small spatial footprint or are varying in location and time, so do not deliver constant and comparable data.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing of the ocean | History

#### History
Remote sensing as we know it today started with the first earth orbiting satellite [Landsat 1] in 1973. Landsat 1 delivered the first multi-spectral images of features on land and coastal zones all over the world and already showed effectiveness in oceanography, although not specifically designed for it. In 1978 [NASA] makes the next step in remote sensing for oceanography with the launch of the first orbiting satellite dedicated to ocean research, [Seasat]. The satellite carried 5 different instruments: a  Radar [altimeter] for retrieving sea surface height, a [microwave] [scatterometer] to retrieve wind speeds and direction, a [microwave] [radiometer] to retrieve [sea surface temperature] (SST), an optical and infrared radiometer to check for clouds and surface characteristics and lastly the first Synthetic Aperture Radar (SAR) instrument. Seasat was only operational for a few months but, together with the [Coastal zone color scanner|Coastal Zone Color Scanner] (CZCS) on [Nimbus 7|Nimbus-7], proved the feasibility of many techniques and instruments in ocean remote sensing. [TOPEX/Poseidon|TOPEX/POSEIDON], an altimeter launched in 1992, provided the first continuous global map of sea surface topography and continued on the possibilities explored by Seasat. The [Jason-1], [OSTM/Jason-2|Jason-2] and [Jason-3] missions continue the measurements from 1992 to today to form a complete time-series of the global sea surface height. Also other techniques hosted on Seasat found continuation. The [Advanced very-high-resolution radiometer|Advanced Very-High-Resolution Radiometer] (AVHRR) Is the sensor carried on al [National Oceanic and Atmospheric Administration|NOAA] missions and made SST retrieval accessible with a continuous time-series since 1979. The [European Space Agency] (ESA) further developed SAR with the [European Remote-Sensing Satellite|ERS-2], [Envisat|ENVISAT] and now [Sentinel-1] missions by providing larger spatial footprints, lowering the resolution and flying [twin missions] to reduce the effective revisit time. Optical remote sensing of the ocean found continuation after the CZCS with [polar orbit]ing missions [Envisat|ENVISAT], [OrbView-2], [Moderate Resolution Imaging Spectroradiometer|MODIS] and very recently with [Sentinel-3], to form a continuous record since 1997. Sentinel-3 is now one of the best equipped missions to map the ocean hosting a SAR altimiter, multispectral spectrometer a radiometer and several other instruments on multiple satellites with alternating orbits providing exceptional temporal and spatial resolution.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing of the ocean | Methods

#### Methods
The physical and chemical state of a surface or object have direct impact on the [emissivity], [reflectance] and [Refraction|refractance] of electromagnetic radiation. Sensors on remote sensing instruments capture radiation, which can be translated back to deduce the physio-chemical properties of the surface. Water content, temperature, roughness and colour are characteristics often deduced from the spectral characteristics of the surface. A sensor on a satellite returns the composite signal for a certain area inside the footprint called a cell, the size of the unique cells is referred to as the spatial resolution. The spatial resolution of a sensor is determined by the distance from earth and the available [Bandwidth (computing)|bandwidth] for data transfer. A satellite passes over the same location consistently through time with the same interval called the revisit-time or temporal resolution. Sensors can not have both a very high temporal and spatial resolution so a tradeoff has to be made specific for the goal of the mission. Sensors on satellites have measuring errors, caused by for example atmospheric interference, geolocation imprecision and topographic distortion. Complete derived products from remote sensing often use simple calculations or algorithms to transform the spectral signature from a cell to a physical value. All methods of transferring spectral data has certain biases which can contribute to the measurement errors of the final result. Often surface characteristics can be deduced with very low error margins due to data corrections, using onboard data or models, and a physically correct translation of spectral characteristics to physio-chemical characteristics.  
Although it is interesting to know the surface characteristics at a certain moment, often research is more interested in documenting the change of a surface over time or the transport of characteristics through space. Change detection leverages the consistent temporal component of remote sensing data to analyze the change of surface properties in time. Change detection relies on having at least two observations taken at different times to analyze the difference between the two images visually or analytically. In land remote sensing change detection is used for example: to assess the impact of a volcano eruption, check the growth of plants through time, map deforestation, and measure ice sheet melt. In oceanography the surface changes more quickly than the revisit time of a satellite making it difficult to monitor certain processes. Change detection in oceanography requires the characteristic to change continuously like [sea level rise] or change spatial scale slower than the revisit time of the satellite like [algal bloom]s. Another way to infer change from only 1 acquisition is by computing the dynamical component and direction from a static image which is leveraged in RADAR altimetry to deduce surface current velocity.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases

### Remote sensing use cases
Radiation scheme showing the main components of incoming radiation for a thermal infrared radiometer. Incoming radiation is either directly emitted by the surface, re-emitted by the atmosphere after absorption, emitted in the atmosphere and reflected at the surface or is reflected sunlight. Only the directly emitted surface radiation gives information so the other noise has to be filtered out using atmospheric correction and cloud detection. reflected sunlight has almost no impact on thermal infrared radiometry.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases | Sea surface temperature (thermal infrared radiometry)

#### Sea surface temperature (thermal infrared radiometry)
The ocean surface emits electromagnetic radiation  B  dependent on the temperature  T  at a certain frequency  \nu  following [Planck's law] for black body radiation, scaled by the emissivity  \epsilon  of the surface since the ocean is not a perfect black body.  
*  B(\nu, T) = \frac{ 2 h \nu^3}{c^2} \frac{1}{e^\frac{h\nu}{k_\mathrm B T} - 1} \cdot \epsilon  
With  B(\nu, T)  the spectral radiance,  h  the  [Planck constant];  c  the [speed of light] and  k  the [Boltzmann constant]. Most radiation emitted by earth is in the [Thermal infrared|thermal-infrared] spectrum which is part of the [Infrared window|atmospheric window], the spectral region for which the atmosphere does not significantly absorb radiation. The radiation coming from the earth's surface with a wavelength within the atmospheric window can be captured by a passive [radiometry] sensor at satellite height. The radiation captured by the sensor is corrected for atmospheric disturbance and radiation noise to compute the [brightness temperature] of the ocean surface. With a correct estimation of the emissivity of sea water (~0.99) the grey body temperature of the ocean surface can be deduced, also referred to as the [Sea surface temperature|Sea Surface Temperature] (SST).  
To correctly remove atmospheric disturbance, both emission and absorption, the airborne radiometers are calibrated for every measurement by SST measurements in multiple bands and/or under different angles. [Atmospheric correction] is only viable if the measured surface is not covered in clouds as they significantly disturb the emitted radiation. Clouds are either removed as viable pixels in the image using cloud busting algorithms or clouds are handled using histogram and spatial coherency techniques (up to 80% cloud cover). Radiometry captures the surface skin temperature (~10 micron depth) of the ocean, which significantly differs from bulk SST in-situ measurements. Phenomena close but not at the surface like diurnal [thermocline] formation are not well captured with satellites but SST can still be of tremendous value in oceanography. Overall satellites measure the SST with a ~0.1-0.6 K accuracy dependent on the sensor and only experience limited issues like surface slicks.  
Retrieved SST datasets really transformed oceanographic research during the 1980's and has multiple different uses. The SST is a clear climatological indicator linking to the [El Niño–Southern Oscillation|ENSO] cycles, [weather] and [climate change] but can also highlight movement of ocean water. SST anomalies can highlight [mesoscale eddies], ocean fronts and regions of upwelling, vertical mixing or river outflow as the water is locally more cold or warm due to transport. The SST is directly linked to the horizontal [density] gradient which is really strong at fronts and is induced by ocean currents and eddies. The currents and fronts are visible in SST images and can be detected using edge detection via [High-pass filter|high pass filters] or kernel transformations to study the dynamics and origin. SST is widely used to track [upwelling] and river outflow strength as these processes are clearly visible as negative SST anomalies.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases | Mapping of algal blooms (Optical)

#### Mapping of algal blooms (Optical)
NDWI, [RGB color model|RGB] and [Normalized difference vegetation index|NDVI] remote sensing image of an [algal bloom] in the San Roque lake in Córdoba [Argentina] derived from [Sentinel-2] level 2a optical data of 2017-02-22. Combining the NDWI and NDVI using thresholding and [edge detection] an image is derived showing a categorized intensity of the algal bloom in the lake. The NDVI image can be combined with in situ measurements or the [spectral signature] of [Chlorophyll a|chlorophyll-a] to make an estimation of the total concentration of phytoplankton/chlorophyll, which is an indication for the pollution of the water.]]
An [Algal bloom|algae bloom] is the enhanced growth of [photosynthetic organism]s in a water system, which manifests itself as a clear change of water color. Algal blooms are often caused by a local enrichment of the water system with nutrients, which temporarily remove the limiting growth factor of photosynthetic organisms like cyanobacteria. Due to oxygen depletion, blocking sunlight and the release of possible toxins algal blooms can be harmful to their environment. Algae are characterized by their green color, caused by the absorption spectra of the [Chlorophyll a|chlorophyll-a] in these organisms. Optical satellites like [Sentinel-2] or active radiometers like [Sentinel-3] and [Moderate Resolution Imaging Spectroradiometer|MODIS] can capture the reflectance of the ocean surface in the visible and near-infrared spectrum. Areas with a higher concentration of algae near the surface have a distinct different color. The spectral signature of an algal bloom in water is captured by the sensor as a high green and near-infrared radiation reflectance and low red light reflectance.  
To map algal blooms thresholding is used in combination with a spectral index like the [Normalized difference vegetation index|Normalized Difference Vegetation Index (NDVI)]. In one observation the intensity and location of the algal bloom can be recorded, and with a second observation at a different time the displacement and intensity change of the algal bloom can be tracked. Algal blooms are used to study internal wave structures, up-welling and river outflows, which all bring nutrients to surface waters, since they are correlated with algae concentration . Pollution often coincides with high nutrient waters, making algal blooms good indicators for the severity and impact of water pollution

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases | Sea surface height ([Satellite geodesy#Altimetry|RADAR altimetry])

#### Sea surface height ([Satellite geodesy#Altimetry|RADAR altimetry])
[Radar|RADAR] altimeters send [microwave] pulses to the surface and catch the reflection intensity over a short time period measuring the two-way travel time of the signal. Electromagnetic radiation travels with the speed of light  c  thus the two way travel time  t  gives information on the height of the satellite above the surface  R_{sat}  following the formula  R_{sat} = \frac{c \cdot t}{2}   . To deduce the sea surface height from the satellite height the two-way travel time has to be corrected for dynamical errors, the atmospheric conditions and the local [geoid] height  h_{geoid} . The local change of the sea surface height due to dynamical effects like wind and currents can be expressed using the following formula.  
h_{dyn} = H_{sat} - RH_{sat} - h_{geoid} - h_{tide} - h_{atm}  
*  h_{dyn}  is the dynamic Sea Surface Height (SSH) which changes dependent on wind and current conditions.
*  H_{sat}  is the height of the satellite above the [reference ellipsoid] which is in the order of a 100&nbsp;km and is known with a cm precision.
*  R_{sat}  is the height of the satellite above the ocean surface and is the quantity measured by the satellite with cm precision.
*  h_{geoid}  is the local height difference between the geoid and the reference ellipsoid which is in the order of ±100&nbsp;cm and can be estimated using a time series of ocean altimetry data.
*  h_{tide}  is the local height difference due to tidal movements of which the magnitude scales dependent of the time of day.
*  h_{atm}  is the local height difference due to atmospheric pressure difference above the surface.
It is hard to correctly estimate  h_{geoid}  ,  h_{tide}  and  h_{atm}  for a certain moment and location. As a solution remote sensing analysts use the Sea Surface Height Anomaly (SSHA) which only requires information on the tidal height and atmospheric pressure, which can be deduced from drifters, weather programs and tidal models. The geoid height for SSHA retrieval is deduced from a long time-series of the same RADAR altimetry data. The SSHA is computed by subtracting the temporal mean of the SSH or Mean Sea Surface (MSS) from the current SSH with  h_{MSS} = [(h_{dyn} + h_{geoid})]  so that:  h_{SSHA} = h_{dyn} + h_{geoid} - MSS  
Although the SSHA can show anomalies in surface currents of the ocean, often a measure called the Absolute Dynamic Topography (ADT) is computed using an independent measurement of the geoid height to display the total ocean currents.  
h_{ADT} = h_{SSHA} + h_{MSS} - h_{geoid}
with the geoid height as a measurement from instruments like the [Gravity and Ocean Circulations Explorer|Gravity and Ocean Circulations Explorer (GOCE)] or  [GRACE and GRACE-FO|Gravity Recovery and Climate Experiment] (GRACE).  
With the launch of [TOPEX/Poseidon|TOPEX/POSEIDON] in 1992 started a continuous time series of global SSH data which, has been extremely valuable in assessing sea level rise in the past decades by combining data with local tide gauges. The dynamical sea surface height from radar altimetry provides useful insight into ocean currents. If assuming [geostrophic balance], the velocity anomaly and direction of surface currents perpendicular to the satellite overpass can be computed using the formula:  
fv_{a} = g \frac{\partial h_{ssha}}{\partial x}   and  fu_{a} = g \frac{\partial h_{ssha}}{\partial y}   for  u = [u] + u_a   and  v = [v] + v_a  
With  f   the Coriolis force,  g   the gravity constant,  u, v   the zonal and meridional velocity and  h_{SSHA}
the derived sea surface height anomaly. RADAR altimeters are able to collect data even in cloudy circumstances but only cover the globe up to latitudes ~60 - 65°. Often the spatial resolution of RADAR altimeters is not too high but their temporal coverage is tremendous, allowing constant monitoring of the ocean surface. RADAR altimeters can also be used to determine the specific wave height and estimate wind velocities using the wave form and backscatter coefficient of the pulse limited return signal.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases | Challenges of Remote Sensing in Coastal Zones

#### Challenges of Remote Sensing in Coastal Zones
Example of MODIS-derived chlorophyll distribution with missing pixels along the coastlines
There can be numerous limitations with the sensors and techniques used by remote sensing tools when it comes to mapping coastal regions. Some challenges stem from issues with resolution and pixel size, as most remote imaging satellites have a pixel size of approximately 1 square kilometer. This presents issues with analyzing coastal regions in the desired level of detail as most coastal processes occur on a spatial scale that is approximately the same (or smaller) than the pixel size provided by remote imaging satellites. Additionally, most ocean sensors have a global coverage frequency of 1–2 days, which may be too long to observe the temporal scale of coastal ocean processes.  
Furthermore, remote sensing of coastal areas has faced challenges in accurately interpreting the color of the ocean. The color of open ocean basins is mostly controlled by phytoplankton and travel predictably or covary with other constituents in the water column like [chlorophyll a]. However, as we get closer to the coastlines and move from the open ocean, to shelf seas, to coastal waters, the particles in the water do not covary with chlorophyll. The apparent color may be influenced by optically active constituents in the water column, such as sediments from runoff or pollution. The satellites can also be influenced by "adjacency effects", where the color of the land can bleed into coastal ocean pixels. Finally, removing the effects of the atmosphere is difficult to achieve because of the complex and dynamic mix of coastal aerosols and sea spray. All of these factors can make it increasingly challenging to accurately analyze coastal regions from remote sensing satellites.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Remote sensing (oceanography) | Remote sensing use cases | Use of UAVs in Remote Sensing

#### Use of UAVs in Remote Sensing
Drone equipped with spectrophotometer.
Satellites, while the core of remote sensing, have limitations in their [Spatial resolution|spatial], [Spectral resolution|spectral], and [Temporal resolution|temporal] resolution. In an effort to combat these limitations, satellite remote sensing utilizes interpolation and modelling to fill in the gaps. While methods of interpolation and modelling can be developed to a high degree of statistical accuracy, they are in their essence an educated guess based on surrounding conditions. The use of UAVs, or drones, as a remote sensing tool can provide data at higher resolutions that can then be used to fill in the gaps in satellite data, often at a lower price than satellites or crewed aircraft. Notable benefits can be found in the pixel gaps found along coastal areas in satellite data as well as the ability to conduct observations of a given area between satellite passes.  
Modern technology has provided UAV users with numerous platforms able to be outfitted with commercial or custom made sensor packages. These sensors consist of [Multispectral imaging|multispectral], [Hyperspectral imaging|hyperspectral] sensors as well as standard visual spectrum, high definition cameras. The size of modern UAVs is also a factor contributing to their applicability. Satellites and crewed aircraft require shore-based facilities or ships capable of supporting take-off and landing operations. Small-UAVs, those defined as under 55 pounds, have the ability to be launched from nearly every location on shore as well as any size vessel at sea. They require very few crew to operate and flight training requirements are affordable and relatively easy to obtain.  
There are some limiting factors to UAV use for oceanic remote sensing. Firstly, the range is limited to the on board fuel or battery capacity as well as distance from the controller. Many governments also impose restrictions on range, stating that UAVs must be flown within unaided visual line of sight. UAV use offshore must be accompanied by a vessel due to these range constraints. Furthermore, the sensors themselves encounter similar challenges to the sensors mounted on satellites, namely in alterations to oceanic reflectance in coastal zones; however, the higher resolution provided by UAV mounted sensors allows for a more diverse assignment of pixels, reducing the blending effect of terrestrial and aquatic environments and reducing the amount of calculations needed to account for reflectance shifts.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial survey | Overview

## Aerial survey  
### Overview  
Aerial survey is a method of collecting geomatics or other imagery data using airplanes, helicopters, UAVs, balloons, or other aerial methods. Typical data collected includes aerial photography, Lidar, remote sensing (using various visible and invisible bands of the electromagnetic spectrum, such as infrared, gamma, or ultraviolet) and geophysical data (such as aeromagnetic surveys and gravity measurements). It can also refer to a chart or map made by analyzing a region from the air. Aerial survey should be distinguished from satellite imagery technologies because of its better resolution, quality, and resistance to atmospheric conditions that can negatively impact and obscure satellite observation. Today, aerial survey is often recognized as a synonym for aerophotogrammetry, a part of photogrammetry where the camera is airborne. Measurements on aerial images are provided by photogrammetric technologies and methods.
Aerial surveys can provide information on many things not visible from the ground.  
Aerial Camera used during WWII for military purposes by the US Army against enemy's submarines
UAV for use in aerial survey applications]]
Pteryx UAV, a civilian for [aerial photography] and photomapping with [Gyro-stabilized camera systems|roll-stabilized] camera head
Aerial survey is a method of collecting [geomatics] or other [image]ry data using [airplane]s, [helicopter]s, [unmanned aerial vehicle|UAV]s, [Balloon (aeronautics)|balloon]s, or other aerial methods. Typical data collected includes [aerial photography], [Lidar], [remote sensing] (using various visible and invisible bands of the [electromagnetic spectrum], such as [infrared], [gamma ray|gamma], or [ultraviolet]) and geophysical data (such as [aeromagnetic survey]s and [gravity] measurements). It can also refer to a chart or map made by analyzing a region from the air. Aerial survey should be distinguished from [satellite imagery] technologies because of its better resolution, quality, and resistance to [Atmosphere of Earth|atmospheric] conditions that can negatively impact and obscure [Earth observation satellite|satellite observation]. Today, aerial survey is often recognized as a synonym for aerophotogrammetry, a part of [photogrammetry] where the [camera] is airborne. Measurements on aerial images are provided by [Photogrammetry|photogrammetric] technologies and methods.  
Aerial surveys can provide information on many things not visible from the ground.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial survey | Terms used in aerial survey

### Terms used in aerial survey
; exposure station or air station: the position of the [Cardinal point (optics)#Principal planes and points|optical center] of the camera at the moment of exposure.
; flying height:  the elevation of the exposure station above the datum (usually mean [sea level]).
;[altitude]: the vertical distance of the aircraft above the [Earth]'s surface.
;tilt the angle between the aerial camera and the horizontal axis perpendicular to the [Steady flight|line of flight].
; tip: the angle between the aerial camera and the line of flight.
;principal point:  the point of intersection of the optical axis of the aerial camera with the [Picture plane|photographical plane].
;isocentre:  the point on the aerial photograph in which the bisector of the angle of tilt meets the photograph.
;[nadir] point: the image of the nadir, i.e. the point on the aerial photograph where a [Plumb bob|plumbline] dropped from the front nodal point pierces the photograph.
;scale: ratio of the [focal length] of the camera objective and the distance of the exposure station from the ground.
;[azimuth]: the clockwise horizontal angle measured about the ground [nadir] point from the ground survey [North] meridian in the plane of photograph.
;[orthomosaic]: A high-resolution map created by orthophotos, usually via drones is termed as an orthomosaic. Ortho meaning a [nadir] image and mosaic meaning a collection of images.
;[Temporal Resolution]:Time between observations.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial survey | Uses

### Uses
Aerial surveys are used for:  
Aerial view of the Paranal Observatory, created by the non-profit initiative [http://wingsforscience.com/ Wings for Science], which offers aerial support to public research organisations  
*[Archaeology]
* [Wild fisheries|Fishery] surveys
* [Geophysics] in [geophysical survey]s
*[Hydrocarbon exploration]
* [Surveying|Land survey]
* [Mining] and [Mining engineering#Pre-mining|mineral exploration]
* [Wildlife observation#Monitoring programs|Monitoring wildlife] and insect populations (called aerial [census] or [Systematic reconnaissance flight|sampling])
* [Environmental monitoring|Monitoring] [vegetation] and [groundcover|ground cover]
* [Reconnaissance]
* [Transport#Infrastructure|Transportation project]s in conjunction with [Surveying|ground survey]s ([Carriageway|roadway], [bridge], [Controlled-access highway|highway])  
Aerial surveys use a measuring camera where the elements of its interior orientation are known, but with much larger [focal length] and [Photographic film|film] and specialized [Camera lens|lens]es.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial survey | Aerial survey sensors

### Aerial survey sensors
In order to carry out an aerial survey, a sensor needs to be fixed to the interior or exterior of the airborne platform with line-of-sight to the target; it is [Remote sensing|remotely sensing]. With manned [aircraft], this is accomplished either through an aperture in the [Skin (aeronautics)|skin] of the aircraft or mounted externally on a wing [strut]. With [unmanned aerial vehicle]s (UAVs), sensors are often mounted under or inside the vehicle, allowing for rapid data collection over challenging terrains, though sometimes with less precision than traditional methods.  
Aerial survey systems typically include the following components:
* Flight navigation software to guide the pilot in flying the desired survey pattern.
* [Satellite navigation|GNSS], combining [Global Positioning System|GPS] and an [inertial measurement unit] (IMU) to provide accurate position and orientation data.
* [Gyro-stabilized camera systems|Gyro-stabilized] mounts to counteract the effects of aircraft roll, pitch, and yaw.
* Data storage units to securely save the recorded data.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial survey | Examples of aerial survey sensors

### Examples of aerial survey sensors
* Vexcel UltraCam today consists of four photogrammetric nadir and oblique cameras (Eagle, Falcon, Osprey, Condor) and their calibrations.
* [Leica Geosystems|Leica] ADS100
* WaldoAir XCAM
* RIEGL LMS-Q780
* [Trimble (company)|Trimble] AX80
* Gpixel GMAX32152 / GMAX32103

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Overview

## Aerial photography  
### Overview  
Aerial photography (or airborne imagery) is the taking of photographs from an aircraft or other airborne platforms. When taking motion pictures, it is also known as aerial videography.
Platforms for aerial photography include fixed-wing aircraft, helicopters, unmanned aerial vehicles (UAVs or "drones"), balloons, blimps and dirigibles, rockets, pigeons, kites, or using action cameras while skydiving or wingsuiting. Handheld cameras may be manually operated by the photographer, while mounted cameras are usually remotely operated or triggered automatically.  
Aerial photography typically refers specifically to bird's-eye view images that focus on landscapes and surface objects, and should not be confused with air-to-air photography, where one or more aircraft are used as chase planes that "chase" and photograph other aircraft in flight.  Elevated photography can also produce bird's-eye images closely resembling aerial photography (despite not actually being aerial shots) when telephotoing from high vantage structures, suspended on cables (e.g. Skycam) or on top of very tall poles that are either handheld (e.g. monopods and selfie sticks), fixed firmly to the ground (e.g. surveillance cameras and crane shots) or mounted above vehicles.  
drone of [Westerheversand Lighthouse], Germany]]
Aerial view of a swimming pool complex
drone of the [Vistula], a river in Poland]]
An aerial view of the city of Pori, Finland
Air photo of a military target used to evaluate the effect of bombing  
Aerial photography (or airborne imagery) is the taking of [photograph]s from an [aircraft] or other [flight|airborne] platforms. When taking [motion picture]s, it is also known as aerial videography.  
Platforms for aerial photography include [fixed-wing aircraft], [helicopter]s, [unmanned aerial vehicle]s (UAVs or "drones"), [balloon (aircraft)|balloons], [blimp]s and [dirigible]s, [rocket]s, [pigeon photography|pigeon]s, [kite aerial photography|kites], or using [action camera]s while [skydiving] or [wingsuiting]. Handheld cameras may be manually operated by the [photographer], while mounted cameras are usually [remote operation|remotely operated] or triggered automatically.
Hraunfossar, [Iceland] captured by a drone-camera
Aerial photography typically refers specifically to [bird's-eye view] images that focus on [landscape]s and [Earth surface|surface] objects, and should not be confused with [air-to-air photography], where one or more aircraft are used as [chase plane]s that "chase" and photograph other aircraft in flight.  [Elevated photography] can also produce bird's-eye images closely resembling aerial photography (despite not actually being aerial shots) when [telephoto]ing from high [overlook|vantage structure]s, [cable-suspended camera system|suspended on cables] (e.g. [Skycam]) or [mast photography|on top of very tall poles] that are either handheld (e.g. [monopod]s and [selfie stick]s), fixed firmly to the ground (e.g. [surveillance camera]s and [crane shot]s) or mounted above [vehicle]s.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | History | Early

#### Early
Honoré Daumier, "Nadar élevant la Photographie à la hauteur de l'Art" (Nadar elevating Photography to Art), published in Le Boulevard, May 25, 1862
Aerial photography was first practiced by the French photographer and [balloonist] [Gaspard-Félix Tournachon], known as [Nadar (photographer)|"Nadar"], in 1858 over [Paris], France. However, the photographs he produced no longer exist and therefore the earliest surviving aerial photograph is titled 'Boston, as the Eagle and the Wild Goose See It.' Taken by [James Wallace Black] and [Samuel Archer King] on October 13, 1860, it depicts [Boston] from a height of 630m.
Equipment Used to Make High-Altitude Photographs (1924)
Aerial view by Cecil Shadbolt, showing Stonebridge Road, [Stamford Hill], and Seven Sisters Curve, part of the [Tottenham and Hampstead Junction Railway], taken from  on 29 May 1882 – the earliest extant aerial photograph taken in the British Isles  
[Kite aerial photography] was pioneered by British meteorologist E.D. Archibald in 1882. He used an explosive charge on a timer to take photographs from the air. The same year, [Cecil Shadbolt] devised a method of taking photographs from the basket of a [gas balloon], including shots looking vertically downwards. One of his images, taken from  over [Stamford Hill], is the earliest extant aerial photograph taken in the British Isles. [Samuel Franklin Cody] developed his advanced 'Man-lifter War Kite' and succeeded in interesting the British [War Office] with its capabilities.  
Antique postcard from Grand Rapids, Michigan, using [Kite aerial photography|kite photo] technique ()  
In 1908, [Albert Samama Chikly] filmed the first ever aerial views using a balloon between [Hammam-Lif] and [Grombalia].
The first use of a motion picture camera mounted to a heavier-than-air aircraft took place on April 24, 1909, over Rome in the 3:28 silent film short, [Wilbur Wright und seine Flugmaschine].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | History | World War I

#### World War I
Giza pyramid complex, photographed from [Eduard Spelterini]'s balloon on November 21, 1904
The use of aerial photography rapidly matured during the war, as [reconnaissance aircraft] were equipped with cameras to record enemy movements and defenses. At the start of the conflict, the usefulness of aerial photography was not fully appreciated, with reconnaissance being accomplished with map sketching from the air.  
Germany adopted the first aerial camera, a [Goerz (company)|Görz], in 1913. The French began the war with several squadrons of [Blériot Aéronautique|Blériot] [Air observation post|observation aircraft] equipped with cameras for reconnaissance. The French Army developed procedures for getting prints into the hands of field commanders in record time.  
[Frederick Charles Victor Laws] started aerial photography experiments in 1912 with No.1 Squadron of the [Royal Flying Corps] (later [No. 1 Squadron RAF]), taking photographs from the British dirigible [British Army airship Beta|Beta]. He discovered that vertical photos taken with a 60% overlap could be used to create a [stereoscopic] effect when viewed in a stereoscope, thus creating a perception of depth that could aid in cartography and in intelligence derived from aerial images. The Royal Flying Corps recon pilots began to use cameras for recording their observations in 1914 and by the [Battle of Neuve Chapelle] in 1915, the entire system of German trenches was being photographed. In 1916, the Austro-Hungarian Monarchy made vertical camera axis aerial photos above Italy for map-making.  
The first purpose-built and practical aerial camera was invented by Captain [John Moore-Brabazon] in 1915 with the help of the [Thornton-Pickard] company, greatly enhancing the efficiency of aerial photography. The camera was inserted into the floor of the aircraft and could be triggered by the pilot at intervals. Moore-Brabazon also pioneered the incorporation of stereoscopic techniques into aerial photography, allowing the height of objects on the landscape to be discerned by comparing photographs taken at different angles.  
By the end of the war, aerial cameras had dramatically increased in size and [optical power|focal power] and were used increasingly frequently as they proved their pivotal military worth; by 1918, both sides were photographing the entire front twice a day and had taken over half a million photos since the beginning of the conflict. In January 1918, [General Allenby] used five Australian pilots from [No. 1 Squadron RAAF|No. 1 Squadron AFC] to photograph a  area in [Palestine (region)|Palestine] as an aid to correcting and improving maps of the Turkish front. This was a pioneering use of aerial photography as an aid for [cartography]. Lieutenants [Leonard Taplin], [Allan Brown (aviator)|Allan Runciman Brown], H. L. Fraser, [Edward Patrick Kenny], and L. W. Rogers photographed a block of land stretching from the Turkish front lines  deep into their rear areas. Beginning 5 January, they flew with a fighter escort to ward off enemy fighters. Using [Royal Aircraft Factory BE.12] and [Martinsyde] airplanes, they not only overcame enemy air attacks, but also had to contend with  winds, antiaircraft fire, and malfunctioning equipment to complete their task.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | History | Commercial

#### Commercial
New York City in 1932, aerial photograph of Fairchild Aerial Surveys Inc
Milton Kent with his aerial camera, June 1953, Milton Kent Studio, Sydney
The first commercial aerial photography company in the UK was [Aerofilms] Ltd, founded by World War I veterans Francis Wills and [Claude Graham White] in 1919. The company soon expanded into a business with major contracts in Africa and Asia as well as in the UK. Operations began from the [Stag Lane Aerodrome] at Edgware, using the aircraft of the London Flying School. Subsequently, the [Aircraft Manufacturing Company] (later the [De Havilland Aircraft Company]), hired an [Airco DH.9] along with pilot entrepreneur [Alan Cobham].  
From 1921, Aerofilms carried out vertical photography for survey and mapping purposes. During the 1930s, the company pioneered the science of [photogrammetry] (mapping from aerial photographs), with the [Ordnance Survey] amongst the company's clients. In 1920, the Australian [Milton Kent] started using a half-plate oblique aero camera purchased from [Carl Zeiss AG] in his aerial photographic business.  
Another successful pioneer of the commercial use of aerial photography was the American [Sherman Fairchild] who started with his own aircraft firm [Fairchild Aircraft] to develop and build specialized aircraft for high altitude [aerial survey] missions. One Fairchild aerial survey aircraft in 1935 carried a unit that combined two synchronized cameras. Utilizing two units of ten lenses each with a ten-inch lens, the aircraft took photos from 23,000 feet. Each photo covered two hundred and twenty-five square miles. One of its first government contracts was an aerial survey of New Mexico to study soil erosion. A year later, Fairchild introduced a better high altitude camera with a nine-lens in one unit that could take a photo covering 600 square miles with each exposure from 30,000 feet.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | History | World War II

#### World War II
Sidney Cotton's [Lockheed 12]A, in which he made a high-speed reconnaissance flight in 1940
In 1939, [Sidney Cotton] and [Flying Officer] [Maurice Longbottom (RAF officer)|Maurice Longbottom] of the [Royal Air Force|RAF] were among the first to suggest that airborne reconnaissance may be a task better suited to fast, small aircraft which would use their speed and high service ceiling to avoid detection and interception. Although this seems obvious now, with modern reconnaissance tasks performed by fast, high flying aircraft, at the time it was radical thinking.  
They proposed the use of [Supermarine Spitfire|Spitfires] with their armament and [radio]s removed and replaced with extra fuel and cameras. This led to the development of the [Supermarine Spitfire (early Merlin powered variants)#PR Mk I – Early Reconnaissance Versions|Spitfire PR] variants. Spitfires proved to be extremely successful in their reconnaissance role and there were many variants built specifically for that purpose. They served initially with what later became [No. 1 Photographic Reconnaissance Unit RAF|No. 1 Photographic Reconnaissance Unit] (PRU). In 1928, the RAF developed an electric heating system for the aerial camera. This allowed reconnaissance aircraft to take pictures from very high altitudes without the camera parts freezing. Based at [RAF Medmenham], the collection and interpretation of such photographs became a considerable enterprise.  
Cotton's aerial photographs were far ahead of their time. Together with other members of the 1 PRU, he pioneered the techniques of high-altitude, high-speed [stereoscopic] photography that were instrumental in revealing the locations of many crucial military and intelligence targets. According to [Reginald Victor Jones|R.V. Jones], photographs were used to establish the size and the characteristic launching mechanisms for both the [V-1 flying bomb] and the [V-2 rocket]. Cotton also worked on ideas such as a prototype specialist reconnaissance aircraft and further refinements of photographic equipment. At the peak, the British flew over 100 reconnaissance flights a day, yielding 50,000 images per day to interpret. Similar efforts were taken by other countries.  
While stationed on an [aircraft carrier] in [Imperial Japan], [FS Hussain], a pilot in the [Royal Indian Air Force], was tasked with photographing the aftermath of the [Atomic bombings of Hiroshima and Nagasaki]. Unaware of the risks of exposure to [radiation poisoning|radiation], it led to his death in 1969 at the age of 44.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Uses

### Uses
Vertical aerial photography is used in [cartography] (particularly in [photogrammetry|photogrammetric] [surveying|surveys], which are often the basis for [topographic maps]), land-use planning, [power line] inspection, [surveillance], construction progress, commercial advertising, [conveyancing], and artistic projects. An example of how aerial photography is used in the field of archaeology is the mapping project done at the site [Angkor Borei and Phnom Da|Angkor Borei] in Cambodia from 1995 to 1996. Using aerial photography, archaeologists were able to identify archaeological features, including 112 water features (reservoirs, artificially constructed pools and natural ponds) within the walled site of Angkor Borei. In the United States, aerial photographs are used in many [Phase I Environmental Site Assessment]s for property analysis.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Aircraft

### Aircraft
In the United States, except when necessary for take-off and landing, full-sized manned aircraft are prohibited from flying at altitudes under 1000 feet over congested areas and not closer than 500 feet from any person, vessel, vehicle or structure over non-congested areas. Certain exceptions are allowed for helicopters, powered parachutes and [Ultralight trike|weight-shift-control aircraft].

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Aircraft | Radio-controlled

#### Radio-controlled
drone technology have allowed aerial photographs to be taken by [quadcopter drone]s, such as this [DJI Mavic] Pro.]]
Advances in [radio controlled model]s have made it possible for [model aircraft] to conduct low-altitude aerial photography. This had benefited [real-estate] advertising, where commercial and residential properties are the photographic subject. In 2014, the US Federal Aviation Administration banned the use of drones for photographs in real estate advertisements. The ban has been lifted and commercial aerial photography using drones of UAS is regulated under the FAA Reauthorization Act of 2018. Commercial pilots have to complete the requirements for a Part 107 license, while amateur and non-commercial use is restricted by the FAA.  
Small scale model aircraft offer increased photographic access to these previously restricted areas. Miniature vehicles do not replace full-size aircraft, as full-size aircraft are capable of longer flight times, higher altitudes, and greater equipment payloads. They are, however, useful in any situation in which a full-scale aircraft would be dangerous to operate. Examples would include the inspection of transformers atop power transmission lines and slow, low-level flight over agricultural fields, both of which can be accomplished by a large-scale radio-controlled helicopter. Professional-grade, gyroscopically stabilized camera platforms are available for use under such a model; a large model helicopter with a 26cc gasoline engine can hoist a payload of approximately . One example is the radio controlled Nitrohawk helicopter developed by [Robert Channon] between 1988 and 1998. In addition to gyroscopically stabilized footage, the use of RC copters as reliable aerial photography tools increased with the integration of FPV (first-person-view) technology. Many radio-controlled aircraft, in particular drones, are now capable of utilizing Wi-Fi to stream live video from the aircraft's camera back to the pilot's or pilot in command's (PIC) ground station.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Regulations | Australia

#### Australia
In Australia, Civil Aviation Safety Regulation Part 101 (CASR Part 101) allows for commercial use of unmanned and remotely piloted aircraft. Under these regulations, unmanned remotely piloted aircraft for commercial are referred to as Remotely Piloted Aircraft Systems (RPAS), whereas radio-controlled aircraft for recreational purposes are referred to as model aircraft. Under CASR Part 101, businesses/persons operating remotely piloted aircraft commercially are required to hold an operator certificate, just like manned aircraft operators. Pilots of remotely piloted aircraft operating commercially are also required to be licensed by the Civil Aviation Safety Authority (CASA).  While a small RPAS and model aircraft may actually be identical, unlike model aircraft, a RPAS may enter controlled airspace with approval, and operate close to an aerodrome.  
Due to a number of illegal operators in Australia, making false claims of being approved, CASA maintains and publishes a list of approved remote operator's certificate (ReOC) holders. However, CASA has modified the regulations and from September 29, 2016, drones under  may be operated for commercial purposes.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Regulations | United States

#### United States  
2006 FAA regulations grounding all commercial RC model flights have been upgraded to require formal FAA certification before permission is granted to fly at any altitude in the US.  
On June 25, 2014, the FAA, in ruling 14 CFR Part 91 [Docket No. FAA–2014–0396] "Interpretation of the Special Rule for Model Aircraft", banned the commercial use of unmanned aircraft over U.S. airspace. On September 26, 2014, the FAA began granting the right to use drones in aerial filmmaking. Operators are required to be licensed pilots and must keep the drone in view at all times. Drones cannot be used to film in areas where people might be put at risk.  
The FAA Modernization and Reform Act of 2012 established, in Section 336, a special rule for model aircraft. In Section 336, Congress confirmed the FAA's long-standing position that model aircraft are aircraft. Under the terms of the Act, a model aircraft is defined as "an unmanned aircraft" that is "(1) capable of sustained flight in the atmosphere; (2) flown within visual line of sight of the person operating the aircraft; and (3) flown for hobby or recreational purposes."  
Because anything capable of being viewed from a [public space] is considered outside the realm of [privacy] in the United States, aerial photography may legally document features and occurrences on private property.  
The FAA can pursue enforcement action against persons operating model aircraft who endanger the safety of the national airspace system: Public Law 112–95, section 336(b).  
On April 7, 2017, the FAA announced special security instructions under 14 CFR § 99.7. Effective April 14, 2017, all UAS flights within 400 feet of the lateral boundaries of U.S. military installations are prohibited unless a special permit is secured from the base and/or the FAA.

Land cover mapping in remote areas with limited existing aerial photography and poor infrastructure <br />(Marshet al. 1994; Slaymaker and Hannah 1997). | Aerial photography | Regulations | United Kingdom

#### United Kingdom  
Aerial photography in the UK has tight regulations as to where a drone is able to fly.  
Aerial Photography on Light aircraft under . Basic Rules for non commercial flying Of a SUA (Small Unmanned Aircraft).  
Article 241 Endangering safety of any person or property states that a person must not recklessly or negligently cause or permit an aircraft to endanger any person or property.  
Article 94 mentions the following about small unmanned aircraft:

Within 50&nbsp;m of any person, during take-off or landing, a small unmanned surveillance aircraft must not be flown within  of any person. This does not apply to the person in charge of the small unmanned surveillance aircraft or a person under the control of the person in charge of the aircraft. | Satellite formation flying | Overview

## Satellite formation flying  
### Overview  
Satellite formation flying is the coordination of multiple satellites to accomplish the objective of one larger, usually more expensive, satellite.  Coordinating smaller satellites has many benefits over single satellites including simpler designs, faster build times, cheaper replacement creating higher redundancy, unprecedented high resolution, and the ability to view research targets from multiple angles or at multiple times.  These qualities make them ideal for astronomy, communications, meteorology, and environmental uses.  
Satellite formation flying is the coordination of multiple [satellites] to accomplish the objective of one larger, usually more expensive, satellite.  Coordinating smaller satellites has many benefits over single satellites including simpler designs, faster build times, cheaper replacement creating higher redundancy, unprecedented high resolution, and the ability to view research targets from multiple angles or at multiple times.  These qualities make them ideal for [astronomy], [communication]s, [meteorology], and [Natural environment|environmental] uses.

Within 50&nbsp;m of any person, during take-off or landing, a small unmanned surveillance aircraft must not be flown within  of any person. This does not apply to the person in charge of the small unmanned surveillance aircraft or a person under the control of the person in charge of the aircraft. | Satellite formation flying | Types of formations

### Types of formations
Landsat-7 being trailed by EO-1 covering the same area at different times
Depending on the application, there are three formations possible: trailing, cluster, and [Satellite constellation|constellation].  
This technology has become more viable thanks to the development of autonomous flying.  With an onboard computer and this algorithm, satellites may autonomously position themselves into a formation.  Previously, [Mission Control Center|ground control] would have to adjust each satellite to maintain formations.  Now, satellites may arrive at and maintain formations with faster response time and have the ability to change the formation for varied resolution of observations.  Also, satellites may be launched from different spacecraft and rendezvous on a particular path.  This advance was made possible by Dave Folta, John Bristow, and Dave Quinn at [NASA]’s [Goddard Space Flight Center] (GSFC).
Cluster formation for the proposed TechSat-21 mission.

Within 50&nbsp;m of any person, during take-off or landing, a small unmanned surveillance aircraft must not be flown within  of any person. This does not apply to the person in charge of the small unmanned surveillance aircraft or a person under the control of the person in charge of the aircraft. | Satellite formation flying | See also

### See also
*[Satellite constellation]
*[Fractionated spacecraft]
*[Magnetospheric Multiscale Mission]

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | Overview

## VisionMap A3 Digital Mapping System  
### Overview  
A3 Digital Mapping System consists of a digital airborne camera and an automatic ground processing system produced by VisionMap. The A3 camera captures imagery using a sweep mechanism, which collects high resolution vertical and oblique imagery simultaneously. The captured data is post-processed by the A3 LightSpeed ground processing system in order to create the final output products. The A3 System is used by numerous national and regional mapping agencies, as well as commercial mapping firms.  
A3 Digital Mapping System consists of a digital [Aerial photography|airborne camera] and an automatic ground processing system produced by VisionMap. The A3 camera captures imagery using a sweep mechanism, which collects high resolution vertical and oblique imagery simultaneously. The captured data is post-processed by the A3 LightSpeed ground processing system in order to create the final output products. The A3 System is used by numerous national and regional mapping agencies, as well as commercial mapping firms.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | A3 Digital Mapping Camera

### A3 Digital Mapping Camera  
The A3 camera consists of dual CCDs with two 300mm lenses mounted on a special motor controlled axis that is attached to the frame, a fast compression and storage unit, and a dual frequency GPS. During flight, sequences of frames are exposed in a cross-track direction at a very high speed, providing a [field of view|FOV] of up to 106 degrees. High resolution vertical and oblique images are captured simultaneously. The camera’s wide field of view maximizes the distance between consecutive flight lines. The camera weighs 38 kilograms, and measures 50x60x60 cm.  
Among the A3 System’s various final products are Super Large Frames (SLF), which are composed of all double frames of one sweep. The size of the SLF is 7,812 pixels along the flight and 62,517 pixels across the flight direction. The typical overlap between two consecutive SLFs in one strip is 30-60%. The overlap between two SLFs in adjacent flight lines is 30-60%. The SLFs may be used for stereo interpretation and stereo [photogrammetry] map production.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | Angular and Movement Stabilization

### Angular and Movement Stabilization  
A3 Digital Mapping Camera utilizes a unique mirror based optical compensation and stabilization method for all movements potentially affecting image quality (forward motion, sweep motion, and general vibrations of the aircraft). The method uses acceleration sensors, GPS data (for aircraft velocity calculation), and the motor encoders for calculation and control of the required motion compensation. Both linear and angular compensations are done by tilting the mirror mounted on the folding optics.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | A3 LightSpeed Processing System

### A3 LightSpeed Processing System  
A3 LightSpeed processing system consists of a PC server configuration and software, which automatically processes data obtained during flight into photogrammetric mapping products. After a pre-process stage in which the user specifies the required outputs and output specifications, the system solves the AT and bundle adjustment automatically. Then a manual geo-referencing stage may be carried out, followed by an automatic processing of DSM, DTM and orthophoto.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | Final Products

### Final Products  
A3 LightSpeed processing system produces the following output products:  
1. Accurately solved vertical and oblique single images of the area of interest. The solution for each image is the result of the photogrammetric bundle block adjustment with self-calibration.
2. SLFs (Super Large Frames), used for stereo photogrammetric mapping.
3. DSM (Digital Surface Model)
4. DTM (Digital Terrain Model)
5. [Orthophoto] with resolution up to the source frame resolution. Global and local radiometric corrections are applied to all frames. The mosaicing process is fully automated and is based on the topographic data (DTM) and radiometric analysis of the area.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | VisionMap A3 Digital Mapping System | Accuracy of Aerial Triangulation

### Accuracy of Aerial Triangulation  
In order to achieve high accuracy, all images and coordinates of projection centers, obtained by GPS, take part in the simultaneous bundle block adjustment with self-calibration. No INS is required.  
As a huge number of single frames are obtained, with significant overlaps between them, a huge amount of common bundles are yielded. This leads to very high redundancy and robustness of the solved system and, therefore, to high final accuracy.  
The A3 System was tested by the [University of Stuttgart] Institute for Photogrammetry (IFP) in their test field. The results of their test are as follows:  
{| class="wikitable"  
! Camera !! Altitude(m) !! GSD(cm) !! RMS East(m) !! RMS North(m) !! RMS Z(m)

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | 3D structure change detection | Overview

## 3D structure change detection  
### Overview  
3D Structure Change Detection is a type of change detection process for GIS (geographical information systems).  It is a process that measures how the volume of a particular area have changed between two or more time periods.  A high-spatial resolution Digital elevation model (DEM) that provides accurate 4-d (space and time) structural information over area of interest is required to compute such changes.  In production, two or more DEMs that cover the same area are used to monitor topographic changes of area.  By comparing the DEMs made at different times, structure of terrain changes can be realized by the ground elevation difference from DEMs.  Details, occurring time and accuracy of such changes are strongly relied on the resolution, quality of DEMs.  In general, the problem of involves whether or not a change has occurred, or whether several changes have occurred.  Such structure changes detection has been widely used to assess urban growth, impact of natural disasters like earthquake, volcano and battle damage assessment.  
3D Structure Change Detection is a type of [change detection (GIS)|change detection] process for [GIS] (geographical information systems).  It is a process that measures how the volume of a particular area have changed between two or more time periods.  A high-spatial resolution [Digital elevation model] (DEM) that provides accurate 4-d (space and time) structural information over area of interest is required to compute such changes.  In production, two or more DEMs that cover the same area are used to monitor topographic changes of area.  By comparing the DEMs made at different times, structure of terrain changes can be realized by the ground elevation difference from DEMs.  Details, occurring time and accuracy of such changes are strongly relied on the resolution, quality of DEMs.  In general, the problem of involves whether or not a change has occurred, or whether several changes have occurred.  Such structure changes detection has been widely used to assess urban growth, impact of natural disasters like earthquake, volcano and battle damage assessment.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | 3D structure change detection | See also

### See also
* [Change detection (GIS)]
* [Digital elevation model]
* [Geographic information system]

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Thermal remote sensing | Overview

## Thermal remote sensing  
### Overview  
Thermal remote sensing is a branch of remote sensing in the thermal infrared region of the electromagnetic spectrum. Thermal radiation from ground objects is measured using a thermal band in satellite sensors.  
Thermal Infrared Image by Mars Odyssey's [thermal emission] imaging system of Mars  
Thermal remote sensing is a branch of [remote sensing] in the [thermal infrared] region of the [electromagnetic spectrum]. [Thermal radiation] from ground objects is measured using a thermal band in [satellite] sensors.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Thermal remote sensing | Principles

### Principles
Thermal remote sensing is working on two major laws which are as follows:
* Identification of geological units and structures
* [Soil moisture] studies
* [Hydrology]
* Coastal zones
* [Volcanology]
* [Forest fire]s: Thermal remote sensing plays a vital role in the determination of [Forest] fire based on the principle of identifying fire [pixel] according to the [temperature] difference between the energy emitting from the surface and ambient temperature.
* [Seismology]
* [Environmental modelling]
* [Meteorology]
* Intelligence / military applications
* Heat loss from buildings

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Thermal remote sensing | Land Surface Temperature (LST)

### Land Surface Temperature (LST)
Cold-air pool on Mt Orjen during the cold-spell in January 2017 shown by a Landsat Land Surface Temperature image. Dolines collected cold air which remained also after sunrise.
One of the most important applications of thermal remote sensing in earth sciences is to calculate the Land Surface Temperature (LST). LST is a measurement of how hot the land is to the touch. It differs from air temperature (the temperature given in weather reports) because land heats and cools more quickly than air. LST is a key variable that is required to accurately model the surface [Earth's energy budget|energy budge]. Thermal [remote sensing] from [satellite]s to derive [land] surface [temperature]s has a long history that can be traced back to the [TIROS-2|TIROS-II] satellite, launched in the early 60s. From the outset certain problems were recognised when deriving temperatures over the land, most notably the low temperatures observed over deserts. To quantify the effects of the atmosphere and the surface ([emissivity] effects) and, both from theory and experiment, various algorithms developed to derive LST.  
[Advanced Spaceborne Thermal Emission and Reflection Radiometer] (ASTER) utilizes a unique combination of wide spectral coverage and high spatial resolution in the visible near-infrared through shortwave infrared to the thermal infrared regions. The ASTER instruments acquire thermal data in Thermal Infrared (TIR) 90 meter Bands (bands 10-14).  
The [Advanced very-high-resolution radiometer|Advanced Very High Resolution Radiometer] (AVHRR) instrument on US [National Oceanic and Atmospheric Administration|National Oceanographic and Atmospheric Administration] (NOAA) 9, 10, 11 and 12 had two bands in [Thermal infrared|Thermal Infrared] regions (bands 4, 5).  
Given recent developments in [Unmanned aerial vehicle|UAVs], thermal images with high spatial and temporal resolutions have become available at a low cost.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Proteus (satellite) | Overview

## Proteus (satellite)  
### Overview  
PROTEUS (acronym for Reconfigurable Platform for Observation, Telecommunications and Scientific Uses) is a 3-axis stabilized platform designed for mini-satellites weighing approximately 500 kg operating in low Earth orbit. The platform is used by six scientific satellites developed as part of the space program of the National Center for Space Studies (CNES) for the European Space Agency: Jason-1, 2 and 3, CALIPSO, CoRoT, and SMOS. The platform is developed by the satellite division of Aérospatiale (in 2016 Thales Alenia Space).  
PROTEUS (acronym for Reconfigurable Platform for Observation, Telecommunications and Scientific Uses) is a 3-axis stabilized platform designed for mini-satellites weighing approximately 500&nbsp;kg operating in [low Earth orbit]. The platform is used by six scientific satellites developed as part of the space program of the [CNES|National Center for Space Studies (CNES)] for the [European Space Agency]: [Jason-1], 2 and 3, [CALIPSO], [CoRoT], and [Soil Moisture and Ocean Salinity satellite|SMOS]. The platform is developed by the satellite division of [Aérospatiale] (in 2016 [Thales Alenia Space]).

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Proteus (satellite) | History

### History
CALIPSO satellite (artist's rendering), 2005
In 1993, CNES decided to launch the development of the PROTEUS mini-satellite program in parallel and jointly with the Jason-1 satellite, the first user for the platform. Program goals including meeting recurring requirements for satellite solutions in the 500&nbsp;kg-700&nbsp;kg class intended for operation in low orbits as platforms for various science and applications. After an industrial consultation with the national prime contractors of the time, Aérospatiale (Cannes) was selected, in May 1996, as industrial prime contractor, with the system to be built in the Cannes-Mandelieu space center.  
By 2010, the PROTEUS platform accumulated 20 years of on-orbit success, with the five satellites that had been orbited: Jason-1, CALIPSO, CoRoT, Jason-2, and SMOS.  
In the same multi-mission platform perspective, the Myriade program to support mission objectives achievable with microsatellites weighing less than 200&nbsp;kg.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Proteus (satellite) | Technical Characteristics

### Technical Characteristics
PROTEUS is a 3-axis stabilized platform designed for missions in low earth orbit for satellites with a total mass of approximately 500&nbsp;kg, including 270&nbsp;kg for the platform excluding propellants. Its main features are as follows:
* Dimensions: 954&nbsp;mm x 954&nbsp;mm x 1,000&nbsp;mm
* Energy: solar panels with an area of 9.5 m2, with one degree of freedom and providing 450 watts (Jason-1) to 550 watts (Jason-3) at the theoretical end of life
* Attitude control: the satellite is stabilized on 3 axes with a pointing precision of 0.15° (Jason-3). The fine sensors used to determine the orientation of the satellite are two three-axis star finder and three two-axis gyrometers. The coarse sensors are three-axis magnetometers and 8 solar sensors with an optical field of 4&nbsp;ft. The orientation is corrected using four reaction wheels which are desaturated using magneto-couplers.
* Propulsion: liquid propellant rocket motors burning hydrazine with a Delta-v capacity of about 120&nbsp;m/s. The mass of hydrazine carried is 28&nbsp;kg (Jason-3).
* Data storage: 500 megabits for telemetry data and 2 gigabits for scientific data.
* Communications: S-band telecommunications with a maximum throughput of 800 kilobits/s.
* Lifespan: 3 years for consumables, 5 years for hardware and radiation tolerance.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Proteus (satellite) | Responsibilities

### Responsibilities
The PROTEUS platform and command and control segment has been developed based on a partnership between CNES and Aérospatiale (now Thales). The integrated team carries out the design of the PROTEUS multi-mission bus, the industrial production of the platform and associated satellites of which is the responsibility of Thales Alenia Space. In accordance with the partnership agreements, CNES remains in contro of workfor its own missions.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Proteus (satellite) | Applications

### Applications
Artists rendering Jason-1 in orbit, 2007
Six satellites use this platform:
* [Jason-1], [remote sensing] for measuring the height of the oceans, launched on December 7, 2001. It celebrated its tenth anniversary in 2011 and reached end of mission in 2013
* [CALIPSO], launched April 28, 2006, meteorological satellite that has been operating for more than 14 years.
* [CoRoT], space telescope for studying the internal structure of stars and searching for exoplanets, launched on December 27, 2006.
* [OSTM/Jason-2], follow-on mission to Jason-1, launched on June 20, 2008.
* [Soil Moisture and Ocean Salinity satellite|SMOS], soil humidity study mission, launched on November 2, 2009.
* [Jason-3], copy of Jason-2, ordered on February 24, 2010 by [EUMETSAT] and placed in orbit in January 2016. This is the most recent satellite using this platform.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | A-train (satellite constellation) | Overview

## A-train (satellite constellation)  
### Overview  
The A-train (from Afternoon Train) is a satellite constellation of three Earth observation satellites of varied nationality in Sun-synchronous orbit at an altitude that is slightly variable for each satellite.
The orbit, at an inclination of 98.14°, crosses the equator each day at around 1:30 pm solar time, giving the constellation its name (the "A" stands for "afternoon") and crosses the equator again on the night side of the Earth, at around 1:30 am.
They are spaced a few minutes apart from each other so their collective observations may be used to build high-definition three-dimensional images of Earth's atmosphere and surface.  
A-train in 2013. As of 2026, the A-Train consists of three satellites. CloudSat and CALIPSO are no longer officially part of the constellation.
The A-train (from Afternoon Train) is a [satellite constellation] of three [Earth observation satellite]s of varied nationality in [Sun-synchronous orbit] at an [altitude] that is slightly variable for each satellite.  
The orbit, at an [Orbital inclination|inclination] of 98.14°, crosses the equator each day at around 1:30 pm [solar time], giving the constellation its name (the "A" stands for "afternoon") and crosses the equator again on the night side of the Earth, at around 1:30 am.  
They are spaced a few [minute]s apart from each other so their collective observations may be used to build high-definition [Three-dimensional space|three-dimensional] images of [Atmosphere of Earth|Earth's atmosphere] and surface.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | A-train (satellite constellation) | Satellites | Active

### Satellites  
#### Active
A-train and C-train in 2019
The train, , consists of three active satellites:
* [OCO-2], lead spacecraft in formation, replaces the failed OCO and was launched for [NASA] on July 2, 2014.
* [GCOM|GCOM-W1 "SHIZUKU"], follows OCO-2 by 11 minutes, launched by [JAXA] on May 18, 2012.
* [Aura (satellite)|Aura], a multi-national satellite, lags OCO-2 by 19 minutes, launched for NASA on July 15, 2004.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | A-train (satellite constellation) | Satellites | Past

#### Past
* [PARASOL (satellite)|PARASOL], launched by CNES on December 18, 2004 and moved to another (lower) orbit on December 2, 2009. PARASOL was deactivated in 2013
* [CloudSat], launched with CALIPSO on April 28, 2006 and moved to another (lower) orbit on February 22, 2018. It was then part of the C-train with Cloudsat until it was officially decommissioned on August 1, 2023.
* [Aqua (satellite)|Aqua], used to run 4 minutes behind GCOM-W1, launched for [NASA] on May 4, 2002. In January 2022, it descended from the A-Train to save fuel and now is in a free-drift mode, wherein its equatorial crossing time is slowly drifting to later times.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | A-train (satellite constellation) | Satellites | Failed

#### Failed
* [Orbiting Carbon Observatory|OCO], destroyed by a launch vehicle failure on February 24, 2009, and was replaced by OCO-2.
* [Glory (satellite)|Glory], failed during launch on a Taurus XL rocket on March 4, 2011, and would have flown between CALIPSO and Aura.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Overview

## Imagery intelligence  
### Overview  
Imagery intelligence (IMINT), pronounced as either as Im-Int or I-Mint, is an intelligence gathering discipline wherein imagery is analyzed (or "exploited") to identify information of intelligence value. Imagery used for defense intelligence purposes is generally collected via satellite imagery or aerial photography.
As an intelligence gathering discipline, IMINT production depends heavily upon a robust intelligence collection management system. IMINT is complemented by non-imaging MASINT electro-optical and radar sensors.  
Fifth Air Force photographic analyst searches for the location of enemy [flak] batteries to plan attacks against enemy positions during the [Korean War].  
Imagery intelligence (IMINT), pronounced as either as Im-Int or I-Mint, is an [list of intelligence gathering disciplines|intelligence gathering discipline] wherein [image]ry is analyzed (or "exploited") to identify [intelligence assessment|information of intelligence value]. Imagery used for [Military Intelligence|defense intelligence] purposes is generally collected via [satellite imagery] or [aerial photography].  
As an intelligence gathering discipline, IMINT production depends heavily upon a robust [intelligence collection management] system. IMINT is complemented by non-imaging [MASINT] electro-optical and radar sensors.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | History | Origins

### History  
#### Origins
> Main: Aerial reconnaissance#History  
> See also: Aerial photography#History  
Sidney Cotton's [Lockheed 12]A, in which he made a high-speed reconnaissance flight in 1940.
Although [aerial photography] was first used extensively in the [Aerial reconnaissance in World War I|First World War], it was only in the [Aerial reconnaissance in World War II|Second World War] that specialized imagery intelligence operations were initiated. High quality images were made possible with a series of innovations in the decade leading up to the war. In 1928, the [RAF] developed an electric heating system for the aerial camera. This allowed reconnaissance aircraft to take pictures from very high altitudes without the camera parts freezing.  
In 1939, [Sidney Cotton] and [Flying Officer] [Maurice Longbottom (RAF officer)|Maurice Longbottom] of the [Royal Air Force|RAF] suggested that airborne reconnaissance may be a task better suited to fast, small aircraft which would use their speed and high service ceiling to avoid detection and interception. They proposed the use of [Supermarine Spitfire|Spitfires] with their armament and [radio]s removed and replaced with extra fuel and cameras. This led to the development of the [Supermarine Spitfire (early Merlin powered variants)#PR Mk I - Early Reconnaissance Versions|Spitfire PR] variants. These planes had a maximum speed of 396&nbsp;mph at 30,000 feet with their armaments removed, and were used for photo-reconnaissance missions. The aircraft were fitted with five cameras which were heated to ensure good results.  
The systematic collection and interpretation of the huge amounts of aerial reconnaissance intelligence data soon became imperative. Beginning in 1941, [RAF Medmenham] was the main interpretation centre for photographic reconnaissance operations in the [European theatre of World War II|European] and [Mediterranean and Middle East theatre of World War II|Mediterranean] theatres. The [Central Interpretation Unit] (CIU) was later amalgamated with the Bomber Command Damage Assessment Section and the Night Photographic Interpretation Section of No 3 Photographic Reconnaissance Unit, [RAF Oakington], in 1942.  
During 1942 and 1943, the CIU gradually expanded and was involved in the planning stages of practically every operation of the war, and in every aspect of intelligence. In 1945, daily intake of material averaged 25,000 negatives and 60,000 prints. Thirty-six million prints were made during the war. By [VE-day], the print library, which documented and stored worldwide cover, held 5,000,000 prints from which 40,000 reports had been produced. According to [Reginald Victor Jones|R.V. Jones], photographs were used to establish the size and the characteristic launching mechanisms for both the [V-1 flying bomb] and the [V-2 rocket].

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | History | Post war spyplanes

#### Post war spyplanes
U-2)]]
Immediately after World War II, long range aerial reconnaissance was taken up by adapted jet bombers – such as the [English Electric Canberra], and its American development, the [Martin B-57] – capable of flying higher or faster than the enemy.  
Highly specialized and secretive strategic reconnaissance aircraft, or spy planes, such as the [Lockheed U-2] and its successor, the [SR-71 Blackbird] were developed by the [United States]. Flying these aircraft became an exceptionally demanding task, as much because of the aircraft's extreme speed and altitude as the risk of being captured as [spy|spies]. As a result, the crews of these aircraft were invariably specially selected and trained.  
There are claims that the US constructed a [hypersonic] reconnaissance aircraft, dubbed the [SR-91 Aurora|Aurora], in the late 1980s to replace the Blackbird. Since the early 1960s, in the United States aerial and satellite reconnaissance has been coordinated by the [National Reconnaissance Office].

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | History | Early use of satellites

#### Early use of satellites
Serum and Vaccine Institute in Al-A'amiriya, Iraq, as imaged by a US reconnaissance satellite in November 2002.
Early photographic reconnaissance satellites used photographic film, which was exposed on-orbit and returned to earth for developing.  These satellites remained in orbit for days, weeks, or months before ejecting their film-return vehicles, called "buckets". Between 1959 and 1984 the U.S. launched around 200 such satellites under the codenames [Corona (satellite)|CORONA] and [KH-7 Gambit|GAMBIT], with ultimate photographic resolution (ground-resolution distance) better than . The first successful mission concluded on 1960-08-19 with the [mid-air recovery] by a [C-119] of film from the Corona mission code-named [Discoverer 14]. This was the first successful recovery of film from an orbiting satellite and the first aerial recovery of an object returning from Earth orbit. Because of a tradeoff between area covered and ground resolution, not all reconnaissance satellites have been designed for high resolution; the [KH-5]-ARGON program had a ground resolution of 140 meters and was intended for [cartography|mapmaking].  
Between 1961 and 1994 the USSR launched perhaps 500 [Zenit spy satellite|Zenit] film-return satellites, which returned both the film and the camera to earth in a pressurized capsule.  
The U.S. [KH-11] series of satellites, first launched in 1976, was made by  [Lockheed Corporation|Lockheed], the same contractor who built the [Hubble Space Telescope].  HST has a 2.4 metre telescope mirror and is believed to have had a similar appearance to the KH-11 satellites. These satellites used [charge-coupled devices], predecessors to modern digital cameras, rather than film. Russian reconnaissance satellites with comparable capabilities are named [Resurs DK] and [Persona (satellite)|Persona].

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Aircraft

### Aircraft
US Navy sailor examining aerial reconnaissance imagery on a light table, 2004.
Low- and high-flying planes have been used all through the last century to gather intelligence about the enemy. U.S. high-flying reconnaissance planes include the [Lockheed U-2], and the much faster [SR-71 Blackbird], (retired in 1998). One advantage planes have over satellites is that planes can usually produce more detailed photographs and can be placed over the target more quickly, more often, and more cheaply, but planes also have the disadvantage of possibly being intercepted by aircraft or missiles such as in the [1960 U-2 incident].  
[Unmanned aerial vehicle]s have been developed for imagery and signals intelligence. These drones are a [force multiplier] by giving the battlefield commander an "eye in the sky" without risking a [aviator|pilot].

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Satellite

### Satellite
Though the resolution of satellite photographs, which must be taken from distances of hundreds of kilometers, is usually poorer than photographs taken by [aerial photography|air], satellites offer the possibility of coverage for much of the earth, including hostile territory, without exposing human pilots to the risk of being shot down.  
Ground-resolution distance achieved by KH-8
There have been hundreds of [reconnaissance satellite]s launched by dozens of nations since the first years of space exploration. Satellites for imaging intelligence were usually placed in high-inclination [low Earth orbit]s, sometimes in [Sun-synchronous orbit]s. Since the film-return missions were usually short, they could indulge in orbits with low [perigee]s, in the range of 100–200&nbsp;km, but the more recent CCD-based satellites have been launched into higher orbits, 250–300&nbsp;km perigee, allowing each to remain in orbit for several years. While the exact [Optical resolution|resolution] and other details of modern [spy satellite]s are classified, some idea of the trade-offs available can be made using simple physics. The formula for the highest possible resolution of an optical system with a circular aperture is given by the [Angular resolution#The_Rayleigh_criterion|Rayleigh criterion]:  
:  \sin \theta = 1.22 \frac{\lambda}{D}.  
Using  
:  \sin \theta =  \frac{\text{size}}{\text{distance}},  
we can get  
:  \text{size} =  1.22\frac{\lambda}{D} \text{distance},  
where θ is the angular resolution, λ is the [wavelength] of light, and D is the diameter of the lens or mirror.  Were the [Hubble Space Telescope], with a 2.4 m telescope, designed for photographing Earth, it would be diffraction-limited to resolutions greater than 16&nbsp;cm (6&nbsp;inches) for green light (  \lambda \approx  550  nm) at its orbital altitude of 590&nbsp;km. This means that it would be impossible to take photographs showing objects smaller than 16&nbsp;cm with such a telescope at such an altitude.  Modern U.S. IMINT satellites are believed to have around 10&nbsp;cm resolution; contrary to references in popular culture, this is sufficient to detect any type of vehicle, but not to read the headlines of a newspaper.  
The primary purpose of most spy satellites is to monitor visible ground activity. While [Image resolution|resolution] and clarity of images has improved greatly over the years, this role has remained essentially the same. Some other uses of satellite imaging have been to produce detailed 3D maps for use in operations and missile guidance systems, and to monitor normally invisible information such as the growth levels of a country's crops or the heat given off by certain facilities. Some of the multi-spectral sensors, such as thermal measurement, are more [electro-optical MASINT] than true IMINT platforms.  
To counter the threat posed by these "eyes in the sky", the [United States], [Soviet Union|USSR]/[Russia], [China] and [India] have developed [anti-satellite weapon|systems for destroying enemy spy satellites] (either with the use of another 'killer satellite', or with some sort of Earth- or air-launched missile).  
Since 1985, commercial vendors of [satellite imagery] have entered the market, beginning with the French [SPOT (satellites)|SPOT] satellites, which had resolutions between 5 and 20 metres. Recent high-resolution (4–0.5 metre) private imaging satellites include [TerraSAR-X], [IKONOS], [Orbview], [QuickBird] and [Worldview-1], allowing any country (or any business for that matter) to buy access to satellite images.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Analytical Methodology

### Analytical Methodology
The value of IMINT reports are determined on a balance between the timeliness and robustness of the intelligence product. As such, the fidelity of intelligence that may be gleaned from imagery analysis is a traditionally perceived by intelligence professionals as a function of the amount of time an imagery analyst (IA) has to exploit a given image or set of imagery. As such, the [United States Army] field manual breaks IMINT analysis into three distinct phases, based upon the amount of time expended in exploiting any given image.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Analytical Methodology | First phase

#### First phase
First phase imagery analysis is deemed "time-dominant". This means that given imagery must be rapidly exploited in order to satisfy an immediate requirement for imagery-sourced intelligence from which a leader may make an educated political and/or military decision. Due to the need to produce near-real time intelligence assessments based upon collected imagery, first phase imagery analysis is rarely compared to collateral intelligence.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Analytical Methodology | Second phase

#### Second phase
Second phase imagery analysis is centered on the further exploitation of recently collected imagery to support short- to mid-term decision-making. Like first phase imagery analysis, second phase imagery analysis is generally catalyzed by a local commander's Priority Intelligence Requirements, at least in the context of a military operational setting. Whereas first phase imagery analysis may depend on the exploitation of a relatively small repository of imagery, or even a single image, second phase imagery analysis generally mandates a review of a chronological set of imagery over time, so as to establish a temporal understanding of objects and/or activities of interest.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | Analytical Methodology | Third phase

#### Third phase
Third phase imagery analysis is generally conducted in order to satisfy strategic intelligence questions or to otherwise explore existing data in the search of "discovery intelligence". Third phase imagery analysis hinges on the use of a large repository of historical imagery as well as access to a variety of sources of information. Third phase imagery analysis incorporates supporting information and intelligence from other [list of intelligence gathering disciplines|intelligence gathering disciplines] and is, therefore, generally conducted in support of a multi-source intelligence team. The exploitation of imagery at this level of analysis is typically conducted with the intention of producing [GEOINT|Geospatial Intelligence] (GEOINT).

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Imagery intelligence | See also

### See also
* [Arthur C. Lundahl]
* [Canadian Forces Intelligence Command|Canadian Forces Joint Imagery Centre] (Canadian GEOINT organization)
* [Defence Imagery and Geospatial Organisation] (DIGO) (Australian GEOINT organization)
* [Defence Intelligence Fusion Centre] (British GEOINT organization)
* [Dino Brugioni|Dino A. Brugioni]
* [First images of Earth from space]
* [Geographic information systems in geospatial intelligence|GIS in GEOINT]
* [Geospatial intelligence] (GEOINT)
* [National Collection of Aerial Photography] (NCAP)
* [National Geospatial-Intelligence Agency] (American GEOINT organization)
* [RAF Intelligence]: [Royal Air Force] Intelligence Branch
* [Remote Sensing]
* [Sentient (intelligence analysis system)]

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Earth observation satellites transmission frequencies | Overview

## Earth observation satellites transmission frequencies  
### Overview  
The earth is constantly monitored by several satellites operating in the earth exploration-satellite service (EESS) or space research service (SRS). These artificial satellites have onboard space radio stations from which they gather data. The data is transmitted back to earth via feeder links. This article lists a number of current active Earth observation satellites and their downlink transmission frequencies.  
> Further: earth exploration-satellite service  
The [earth] is constantly monitored by several [satellite]s operating in the earth exploration-satellite service (EESS) or [space research service] (SRS). These artificial satellites have onboard [space radio station]s from which they gather [data]. The data is transmitted back to earth via [feeder link]s. This article lists a number of current active [Earth observation satellite]s and their [downlink] transmission frequencies.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Earth observation satellites transmission frequencies | Frequency assignments

### Frequency assignments
{| class="wikitable" style="font-size:98%;"  
! Satellite
! Frequency
!align="center" colspan="2"| Band

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Overview

## Advisory Committee for Earth Observation  
### Overview  
The Advisory Committee for Earth Observation (ACEO) is the senior advisory body to the European Space Agency's (ESA's) Director of Earth Observation (EO) Programmes.  
The Advisory Committee for Earth Observation (ACEO) is the senior advisory body to the [European Space Agency|European Space Agency's] (ESA's) Director of Earth Observation (EO) Programmes.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Role

### Role
ACEO is the main interpreter of the views and needs of the European scientific community as regards access to space experimentation and EO data exploitation in the Earth science community.  Its main tasks are to advise and/or make recommendations on:  
* the needs of the scientific community for spaceborne observations of Earth;
* scientific activities during the development, implementation and exploitation of the approved projects of ESA's Earth Observation programmes
* the formulation and updating of medium and long-term Earth Observation Science Strategy in Europe in regard to the interests of the scientific community;
* the priorities of the scientific community in the selection (and extension) and formulation of [future Earth] Observation missions, and the associated [https://EOPRO.esa.int Calls for Proposals for new mission ideas];
* the science aspects in relation to the evolution of operational Earth Observation missions in terms of the [space segment] (planning, succession, Long-Term Scenario), the operational services and the impact of operational missions on EO Science;
* the contribution of ESA EO missions to address major societal issues;
* the scientific studies and activities required to lay the foundations for future missions;
* the selection of new scientific projects.  
The ACEO has the highest-level advisory capacity, through the Director of Earth Observation Programmes (D/EOP), for matters to be treated at the level of the Programme Board for Earth Observation (PB-EO), which may also request advice on particular issues of a scientific nature.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Membership

### Membership
ACEO is composed of invited members who are senior European scientific experts and ex-officio representatives of relevant stakeholder bodies, and is managed by the ACEO Secretariat in the [Mission Science Division|Earth and Mission Science Division]. ACEO members are selected ESA Executive to achieve diversity and balance in scientific expertise across the relevant Earth science disciplines, from within ESA's 23 Member States and Cooperating States.  
Membership appointments are for a period of three years with the possibility to extend to a maximum of five years. The tenure of the chair is for three years, independent of previous normal membership, with the possibility of a one-year extension.  
The Chair of the PB-EO is invited, ex officio, to attend the meetings of the ACEO. Furthermore, representatives of other ESA Directorates’ Science Advisory Committees may be invited to attend.  
Representatives of scientific or professional organisations with a stake in Earth Observation (e.g. [http://essc.esf.org/ European Space Sciences Committee (ESSC)] of the [European Science Foundation], whose expertise may be required by ACEO to properly fulfil its functions can also be invited, ex-officio and Ad Personam, to attend the meetings.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Operating Procedure

### Operating Procedure
The Committee meets at dates and places jointly agreed between the chair and the Director of Earth Observation Programmes.  
ACEO provides advice and/or makes recommendations on the items referred to it by the Director of Earth Observation Programmes. Members of the Committee may however raise issues they wish to discuss.  
The Chair of the ACEO reports on the committee's recommendations at the meetings of the PB-EO. The Chair of PB-EO can make a request to the Director of Earth Observation Programmes for ACEO to carry out specific actions in the framework of its mandate.  
The ACEO Secretariat is managed by ESA's [Mission Science Division|Earth and Mission Science Division].

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Current Members

### Current Members
* [Jonathan Bamber|Prof. Jonathan Bamber] (UK)
* [https://www.femtech.at/expertinnen/datenbank/26636 Dr. Annett Bartsch] (AT)
* [https://sites.units.it/braitenberg/?file=biografia.htm Prof. Carla Braitenberg] (IT)
* [https://www.lmd.polytechnique.fr/~chepfer/ Prof. Helène Chepfer] (FR)
* [https://ara.lmd.polytechnique.fr/index.php?page=cyril-crevoisier Dr. Cyril Crevoisier] (FR)
* [https://eo.belspo.be/en/news/2-stereo-researchers-selected-new-aceo-members Prof. Gabriëlle De Lannoy] (BE)
* [https://geospatialframeworks.com.au/professor-rene-forsberg/ Prof. Rene Forsberg] (DK), ACEO Chair
* [https://satsummit.io/2024-lisbon/speakers/anne-fouilloux/ Dr. Anne Fouilloux] (NO)
* [https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-5P/Vincent-Henri_Peuch_Head_of_Copernicus_Atmosphere_Monitoring_Service Dr. Vincent-Henri Peuch] (DE)
* [https://www.bbaw.de/die-akademie/bbaw-mitglieder/mitglied-markus-rapp Prof. Markus Rapp] (DE)
* [https://www.uu.se/en/contact-and-organisation/staff?query=N96-3829 Prof. Anna Rutgersson] (SE)
* [https://iris.cnr.it/cris/rp/rp21545 Prof. Rosalia Santoleri] (IT)
* [https://remotesensing.vito.be/team/else-swinnen Dr. Else Swinnen] (BE)
* [Chris Rapley|Prof. Chris Rapley], Ex-officio Member (ESSC Chair)
* [https://www.univ-nantes.fr/olivier-grasset Prof. Olivier Grasset], Ex-officio Member (SSAC Chair)
* [:fi:Jarkko_Koskinen_(professori)|Prof. Jarkko Koskinen], Ex-officio Member (PB-EO Chair)

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | History

### History
The ACEO was established in 2018  to replace the former Earth Science Advisory Committee (ESAC) as the senior advisory body to the Director of Earth Observation.  
The former Earth Science Advisory Committee had been in place operating as the main science advisory body from the inception of the [Living Planet Programme|ESA Living Planet Programme] in 1996 until 2018. Prior to that the Earth Observation Advisory Committee (EOAC), created in 1981, had been the senior scientific advisory body to the Application Satellites Directorate on programmes, projects, studies and application of this field.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Advisory Committee for Earth Observation | Frontiers of Science

### Frontiers of Science
ACEO meetings typically include a "Frontiers of Science" agenda item, where scientific topics are discussed with the broader ESA audience. Past presentations are listed below:  
* ACEO No.01, Martin Visbeck, (Ocean) Sustainability, Innovation and Partnership
* ACEO No.02, Daniel Selva, Architecting Earth Observing Systems: Architecting Earth Observing Systems: An overview and comparison of processes
* ACEO No.03, Maria Brovelli & Jacqui McGlade, EO in Society: Open Science and Innovation
* ACEO No.04, Thomas Blaschke, Space 4.0 and Big Earth data: towards integrated Earth Observation Science
* ACEO No.05, No presentation (Earth Explorer 9 Selection Meeting)
* ACEO No.06, Piet Stammes, TROPOMI on Sentinel-5P: Monitoring atmospheric composition for air quality and climate applications
* ACEO No.07, Johanna Tamminen, Space-based carbon dioxide observations to support Paris Agreement
* ACEO No.08, No presentation (Scout Selection Meeting)
* ACEO No.09, No presentation ( Earth Explorer 10 Candidate Selection Meeting)
* ACEO No.10, Florence Rabier, The Value of Observations in Earth System Modelling
* ACEO No.11, Andrew Watson, Observing the changing ocean carbon cycle
* ACEO No.12, Sonia Senevirvatne, Weather and Climate Extremes in a Changing Climate: Newest evidence and relevance for ESA
* ACEO No.13, René Forsberg, Measuring the melting [cryosphere] from space and airborne missions
* ACEO No.14, No presentation ([https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Preparing_for_tomorrow/Earth_Explorer_User_Consultation_Meetings Earth Explorer 10 User Consultation Meeting])
* ACEO No.15, Nico Sneeuw, Gravitational remote sensing of system Earth
* ACEO No.16, Kathy Whaler, The Earth’s deep interior and satellite magnetometry
* ACEO No.17, Keith Raney, Microwave System Science – Once and Present Frontier
* ACEO No.18, No presentation ([https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Preparing_for_tomorrow/Earth_Explorer_User_Consultation_Meetings Earth Explorer 11 User Consultation Meeting])
* ACEO No.19, No presentation (Earth Explorer 12 Candidate Selection Meeting)
* ACEO No.20, Markus Rapp, The MLT-region: the ignorosphere
* ACEO No.21, Helene Chepfer, The Contribution of Space Lidars to Climate Science
* ACEO No.22, Jonathan Bamber, Mass balance of [The Cryosphere|the cryosphere] and implications for [sea level rise] and closing the sea level budget
* ACEO No.23, No presentation ([https://www.esa.int/Applications/Observing_the_Earth/FutureEO/Preparing_for_tomorrow/Earth_Explorer_User_Consultation_Meetings Earth Explorer 11 User Consultation Meeting])

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Image foresting transform | Overview

## Image foresting transform  
### Overview  
In the practice of digital image processing Alexandre X. Falcao, Jorge Stolfi, and Roberto de Alencar Lotufo have created and proven that the Image Foresting Transform (IFT) can be used as a time saver in processing 2-D, 3-D images, and moving images.  
In the practice of [digital image processing] Alexandre X. Falcao, Jorge Stolfi, and Roberto de Alencar Lotufo have created and proven that the Image Foresting Transform (IFT) can be used as a time saver in processing 2-D, 3-D images, and moving images.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Image foresting transform | Definition

### Definition  
The transform is a tweaked version of Dijkstra’s shortest-path algorithm that is optimized for using more than one input and the maximization of digital image processing operators. The transform makes a graph of the pixels in an image and the connections between these points are the "cost" of the path portrayed. The cost is calculated by inspecting the characteristics, for example, grey scale, color, gradient among many others, of the path between pixels. Trees are made by connecting the pixels that have the same or close cost for applying the operator decided upon. The robustness of the transform does come at a cost and uses a lot of storage space for the code and the data being processed. When the transform is through, the predecessor, cost, and label are returned. Most of the operators that are used for digital image processing can use these three pieces of information to be optimized.

Compressing the bright range of input values: This process involves reducing the brightness of the brighter areas in the image to prevent overexposure resulting in a more balanced and visually appealing image. | Satellite Image Time Series | Overview

## Satellite Image Time Series  
### Overview  
A Satellite Image Time Series (SITS) is a set of satellite images taken from the same scene at different times. A SITS makes use of different satellite sources to obtain a larger data series with short time interval between two images. In this case, it is fundamental to observe the spatial resolution and registration constraints.
Satellite observations offer opportunities for understanding how Earth is changing, for determining the causes of these changes, and for predicting future changes. Remotely sensed data, combined with information from ecosystem models, offers an opportunity for predicting and understanding the behavior of the Earth's ecosystem. Sensors with high spatial and temporal resolutions make the observation of precise spatio-temporal structures in dynamic scenes more accessible. Temporal components integrated with spectral and spatial dimensions allow the identification of complex patterns concerning applications connected with environmental monitoring and analysis of land-cover dynamics.  
A Satellite Image Time Series (SITS) is a set of [satellite image]s taken from the same scene at different times. A SITS makes use of different satellite sources to obtain a larger data series with short time interval between two images. In this case, it is fundamental to observe the [spatial resolution] and registration constraints.  
Satellite observations offer opportunities for understanding how [Earth] is changing, for determining the causes of these changes, and for predicting future changes. Remotely sensed data, combined with information from [ecosystem] models, offers an opportunity for predicting and understanding the behavior of the Earth's [ecosystem]. Sensors with high spatial and [temporal resolution]s make the observation of precise spatio-temporal structures in dynamic scenes more accessible. Temporal components integrated with [Electromagnetic spectrum|spectral] and spatial dimensions allow the identification of complex patterns concerning applications connected with [environmental monitoring] and analysis of land-cover dynamics.

Should be able to access this information in a way that reduces, or ideally eliminates, any barriers to viewing and interpreting it. | Aerial archaeology | History | O.G.S. Crawford and the systematic use of aerial photography

#### O.G.S. Crawford and the systematic use of aerial photography
In the 1920s, Osbert Guy Stanhope Crawford emerged as a key figure in the development of aerial archaeology. Crawford recognized the potential of aerial photography for systematically documenting archaeological sites. His work in Britain and the Middle East demonstrated that aerial surveys could reveal features such as crop marks and [Soil mark|soil disturbances] that were undetectable from ground level. His pioneering efforts helped establish aerial archaeology as a legitimate and essential method within the broader field of archaeological research.  
[George W. G. Allen|Major G. W. G. Allen] was an English engineer who, after learning of the work Crawford was doing, he was inspired to use his own airplane around Southern England, taking photographs of the landscape. His work in documenting [Prehistory|prehistoric] landscapes and [Roman roads] in England marked a significant advancement in the application of aerial [Methodology|methods] for archaeological surveys.

Should be able to access this information in a way that reduces, or ideally eliminates, any barriers to viewing and interpreting it. | Aerial archaeology | History | Modern aerial archaeology | LiDAR and photogrammetry

##### LiDAR and photogrammetry
Technological advancements such as [Lidar|LiDAR (Light Detection and Ranging)] and [photogrammetry] have further enhanced aerial archaeology. LiDAR, in particular, is capable of penetrating dense vegetation to reveal features hidden beneath forest canopies, making it a valuable tool for studying heavily forested regions. Photogrammetry, which allows for the creation of precise 3D models of archaeological sites, has enabled researchers to document and analyze sites with increased speed and accuracy. Together, these technologies have expanded the potential of aerial archaeology, allowing for more detailed and comprehensive analyses of archaeological sites.  
Notable people in the field of aerial archaeology include:  
* [O. G. S. Crawford] ([Great Britain|England]): Pioneered the use of aerial photography for archaeological survey.
* [Roger Agache] (France):  A leading figure in the development of aerial archaeology in France, known for his work on Gallo-Roman sites.
* [Antoine Poidebard] ([Syria]): Used aerial photography to study Roman [Limes (Roman Empire)|limes] in the Middle East.
* [Dache McClain Reeves] (United States): A pioneer in the use of aerial photography to study Native American archaeology in the United States.
* [Henry Wellcome] ([List of archaeological sites by country#Sudan|Sudan])
* [Lionel Rees] ([List of archaeological sites by country#Jordan|Jordan])
* [Giacomo Boni (archaeologist)|Giacomo Boni] ([List of archaeological sites by country#Italy|Italy])

Should be able to access this information in a way that reduces, or ideally eliminates, any barriers to viewing and interpreting it. | Aerial archaeology | Techniques and technologies | Aerial photography

### Techniques and technologies  
#### Aerial photography
Photography is the most common method used in aerial archaeology.  Archaeologists use specialized cameras and lenses to capture high-resolution images of the ground from aircraft or drones.

Should be able to access this information in a way that reduces, or ideally eliminates, any barriers to viewing and interpreting it. | Aerial archaeology | Techniques and technologies | Remote sensing

#### Remote sensing
Beyond traditional aerial photography, archaeologists use a range of remote sensing techniques to investigate sites without physical excavation. These methods involve collecting data from a distance using specialized sensors that detect and record different forms of [electromagnetic radiation]. This information can reveal subsurface features, variations in vegetation, and other archaeological clues hidden from the naked eye. Digital data, for example, ALS, can be used effectively in "heavily automated workflows," (a process that uses rule-based logic to launch tasks that run without human intervention), e.g. a six-year project using supervised automated classification to survey  of Baden-Wurttemberg in Germany, identified as many as 600,000 possible sites. LANDSAT images have helped in identifying large-scale features, such as an ancient riverbed running from the Saudi Arabian desert to Kuwait.  
[SLAR] (sideways looking airborne radar) is a remote sensing technique that records pulses of [electromagnetic radiation] from an aircraft. Richard Adams used SLAR to identify a matrix of possible Mayan water irrigation systems underneath the dense rainforest from a NASA aircraft.  
[Synthetic-aperture radar|SAR] (synthetic aperture radar) involves radar images that are processed to create high-resolution data. It can be faster and less time-consuming than surface survey.Lynchet system near Bishopstone in 300x300px

Should be able to access this information in a way that reduces, or ideally eliminates, any barriers to viewing and interpreting it. | Aerial archaeology | Techniques and technologies | Satellite imagery

#### Satellite imagery
In places yet to be documented (or where maps are considered confidential), satellite imagery is vital to providing base maps for [Excavations, Archaeological|excavation]. (to be investigated further for a greater understanding). We can thus see the impressive effect aerial methods can have on streamlining archaeological survey, and pushing the limits of what is possible.

For each pixel, label the pixel and form a new [Feature (machine learning)|feature vector] for it.

# For each pixel, label the pixel and form a new [Feature (machine learning)|feature vector] for it.

Use the new feature vector and combine the contextual information to assign the final label to the | Merging the pixels in earlier stages

# Use the new feature vector and combine the contextual information to assign the final label to the  
#### Merging the pixels in earlier stages
Instead of using single pixels, the neighbour pixels can be merged into homogeneous regions benefiting from contextual information. And provide these regions to classifier.

Use the new feature vector and combine the contextual information to assign the final label to the | Acquiring pixel feature from neighbourhood

#### Acquiring pixel feature from neighbourhood
The original spectral data can be enriched by adding the contextual information carried by the neighbour pixels, or even replaced in some occasions. This kind of pre-processing methods are widely used in [Image texture|textured image] recognition. The typical approaches include mean values, variances, texture description, etc.

Use the new feature vector and combine the contextual information to assign the final label to the | Sherman Fairchild | Aerial photography

### Aerial photography
Aerial Age Nov. 7 1921, magazine cover with aerial photo of Columbia University shot by W. L. Hamilton for Fairchild Aerial Camera Corp.  
Fairchild F-1 Aerial Camera  
In 1917, after being rejected from the military because of his poor health, Fairchild was determined to find another way to support the [World War I] effort. To accommodate this growing commercial demand for aerial surveys, Fairchild established Fairchild Aerial Surveys in the United States. In 1965 Fairchild sold Fairchild Aerial Surveys to Aero Services, Inc., which decided to keep only the more recent photographs and dispose of the others. A former Fairchild employee learned of this plan and was able to get the older material to three Southern California Institutions, [Whittier College], [UCLA], and [California State University at Northridge], where he knew professors who would put the material to good use. The [University of California Santa Barbara] acquired the collection in December, 2012.

Use the new feature vector and combine the contextual information to assign the final label to the | Sherman Fairchild | Aerial photography | Lunar photography

#### Lunar photography
The Fairchild Lunar Mapping Camera
Fairchild Corporation developed the Fairchild Lunar Mapping Camera (also known as the Metric Camera) for [NASA]. The camera was carried on [Apollo 15], [Apollo 16|16], and [Apollo 17|17] and took photos from lunar orbit throughout the missions. Over 7,000 frames were captured by the Lunar Mapping Cameras, covering approximately 20% of the lunar surface.

Use the new feature vector and combine the contextual information to assign the final label to the | Collocation (remote sensing) | Overview

## Collocation (remote sensing)  
### Overview  
Collocation is a procedure used in remote sensing
to match measurements from two or more different instruments.
This is done for two main reasons:
for validation purposes when comparing measurements of the same variable,
and to relate measurements of two different variables
either for performing retrievals or for prediction.
In the second case the data is later fed into some type of statistical
inverse method
such as an artificial neural network, statistical classification algorithm,
kernel estimator or a linear least squares.
In principle, most collocation problems can be solved by a nearest neighbor search,
but in practice there are many other considerations involved and the best method is
highly specific to the particular matching of instruments.
Here we deal with some of the most important considerations along with specific examples.
There are at least two main considerations when performing collocations.
The first is the sampling pattern of the instrument.
Measurements may be dense and regular, such as those from a cross-track
scanning satellite instrument.  In this case, some form of interpolation
may be appropriate.  On the other hand, the measurements may be
sparse, such as a one-off field campaign designed for some
particular validation exercise.
The second consideration is the instrument footprint, which
can range from something approaching a point measurement
such as that of a radiosonde, or it might be several
kilometers in diameter such as that of a satellite-mounted,
microwave radiometer.  In the latter case, it is appropriate
to take into account the instrument antenna pattern when
making comparisons with another instrument having both a smaller
footprint and a denser sampling, that is, several measurements
from the one instrument will fit into the footprint of the other.
Just as the instrument has a spatial footprint, it will also have
a temporal footprint, often called the integration time.
While the integration time is usually less than a second,
which for meteorological applications is essentially instantaneous,
there are many instances where some form of time averaging can considerably
ease the collocation process.
The collocations will need to be screened based on both the time
and length scales of the phenomenon of interest.
This will further facilitate the collocation process since
remote sensing and other measurement data is almost always
binned in some way.
Certain atmospheric phenomena such as clouds or convection are quite transient
so that we need not consider collocations with a time error of more than an hour or so.
Sea ice, on the other hand, moves and evolves quite slowly, so that
measurements separated by as much as a day or more might still be useful.  
Collocation is a procedure used in [remote sensing]
to match measurements from two or more different instruments.
This is done for two main reasons:
for validation purposes when comparing measurements of the same variable,
and to relate measurements of two different variables
either for performing retrievals or for prediction.
In the second case the data is later fed into some type of statistical
[inverse problem|inverse method]
such as an [artificial neural network], [statistical classification] algorithm,
[kernel estimation|kernel estimator] or a [Linear least squares (mathematics)|linear least squares].
In principle, most collocation problems can be solved by a [nearest neighbor search],
but in practice there are many other considerations involved and the best method is
highly specific to the particular matching of instruments.
Here we deal with some of the most important considerations along with specific examples.  
There are at least two main considerations when performing collocations.
The first is the sampling pattern of the instrument.
Measurements may be dense and regular, such as those from a cross-track
scanning satellite instrument.  In this case, some form of [interpolation]
may be appropriate.  On the other hand, the measurements may be
sparse, such as a one-off field campaign designed for some
particular validation exercise.
The second consideration is the instrument footprint, which
can range from something approaching a point measurement
such as that of a [radiosonde], or it might be several
kilometers in diameter such as that of a satellite-mounted,
microwave radiometer.  In the latter case, it is appropriate
to take into account the instrument [antenna pattern] when
making comparisons with another instrument having both a smaller
footprint and a denser sampling, that is, several measurements
from the one instrument will fit into the footprint of the other.  
Just as the instrument has a spatial footprint, it will also have
a temporal footprint, often called the integration time.
While the integration time is usually less than a second,
which for meteorological applications is essentially instantaneous,
there are many instances where some form of time averaging can considerably
ease the collocation process.  
The collocations will need to be screened based on both the time
and length scales of the phenomenon of interest.
This will further facilitate the collocation process since
remote sensing and other measurement data is almost always
[data binning|binned] in some way.
Certain atmospheric phenomena such as clouds or convection are quite transient
so that we need not consider collocations with a time error of more than an hour or so.
Sea ice, on the other hand, moves and evolves quite slowly, so that
measurements separated by as much as a day or more might still be useful.

Use the new feature vector and combine the contextual information to assign the final label to the | Collocation (remote sensing) | Satellites

### Satellites  
Polar-stereographic projection showing 12 hours of measurements from three AMSU-B instruments  
The [satellite]s that most concern us are those with a [low Earth orbit|low-Earth], [polar orbit] since [geostationary] satellites view the same point throughout their lifetime.
The diagram shows measurements from [Advanced Microwave Sounding Unit|AMSU-B]
instruments mounted on three satellites over a period of 12 hours.
This illustrates both the orbit path and the scan pattern which runs crosswise.
Since the [orbit] of a satellite is [deterministic system|deterministic],
barring [orbital maneuver|orbit maneuvers], we can predict the location of the
satellite at a given time and, by extension, the location of
the measurement pixels.
In theory, collocations can be performed by inverting the
determining equations starting from the desired time period.
In practice, partially processed data (usually referred to as
level 1b, 1c or level 2) contain the coordinates of each of
the measurement pixels and
it is common to simply feed these coordinates to a nearest neighbor search.
As mentioned previously, the satellite data is always [Data binning|binned]
in some manner.
At minimum, the data will be arranged in
swaths extending from pole to pole.
The swaths will be labelled by time period and the
approximate location known.

Use the new feature vector and combine the contextual information to assign the final label to the | Collocation (remote sensing) | Radiosondes

### Radiosondes  
Ascent of a weather balloon launched from the Polarstern research vessel  
[Radiosonde]s are particularly important for collocation studies
because they measure atmospheric variables more accurately and more
directly than satellite or other remote-sensing instruments.
In addition, radiosonde samples are effectively instantaneous point measurements.
One issue with radiosondes carried aloft by [weather balloon]s is
balloon drift.  In,
this is handled by averaging all the satellite pixels within a 50&nbsp;km radius
of the balloon launch.  
Histogram of ascent rates of weather balloons launch from the Polarstern research vessel  
If high-resolution sonde data, which normally has a constant
sampling rate or includes the measurement time, is used,
then the lateral motion can be traced from the wind data.
Even with low-resolution data, the motion can still
be approximated by assuming a constant ascent rate.
Excepting a short bit towards the end,
the linear ascent can be clearly seen in the figure above.
We can show that the ascent rate of a balloon is given
by the following equation
:  
:
v = \sqrt{\frac{g k h (1 - R_a/R_s)}{c_D}}  
where g is gravitational acceleration,
k relates the height, h, and surface area, A,
of the balloon to its volume: V&nbsp;=&nbsp;khA;
Rs is the equivalent "[gas constant]" of the balloon,
Ra is the gas constant of the air
and cD is the drag coefficient of the balloon.
Substituting some sensible values for each of the constants,
k=1. (the balloon is a perfect cylinder), h=2. m, cD&nbsp;=&nbsp;1.
and Ra is the gas constant of helium,
returns an ascent rate of 4.1&nbsp;m/s.  Compare this with the
values shown in the histogram which compiles all of the
radiosonde launches from the [Polarstern] research vessel
over a period of eleven years between 1992 and 2003.

Use the new feature vector and combine the contextual information to assign the final label to the | Collocation (remote sensing) | Interpolation

### Interpolation  
For gridded data such as [data assimilation|assimilation] or [meteorological reanalysis|reanalysis] data,
[interpolation] is likely the most appropriate method for performing any type of comparison.
A specific point in both physical position and time is easy to locate
within the grid and interpolation performed between the nearest neighbors.
[Linear interpolation] ([bilinear interpolation|bilinear], [trilinear interpolation|trilinear] etc.) is the most common,
though cubic is used as well but is probably not worth the extra computational overhead.
If the variable of interest has a relatively smooth rate of change (temperature is a good example of this because it
has a diffusion mechanism, [radiative transfer], not available to other atmospheric variables),
then interpolation can eliminate much of the error associated with collocation.  
Interpolation may also be appropriate for many types of satellite instruments,
for instance a cross-track scanning instrument like [Landsat].
In  data derived from the [Advanced Microwave Sounding Unit] (AMSU) are
interpolated (although not for the purposes of collocation) using a slight variation
of trilinear interpolation.
Since measurements within a single scan track are laid out in an approximately rectangular
grid, bilinear interpolation can be performed.
By searching for the nearest overlapping scan track both forwards and backwards in time,
the spatial interpolates can then be interpolated in time.
This technique works better with derived quantities rather than raw brightness temperatures since
the scan angle will already have been accounted for.  
For instruments with a more irregular sampling pattern, such as the Advanced Microwave
Scanning Radiometer-EOS (AMSR-E) instrument which has a circular scanning pattern,
we need a more general form of interpolation such as [kernel estimation].
A method commonly used for this particular instrument, as well as [SSM/I],
is a simple daily average within regularly gridded, spatial bins
.

Use the new feature vector and combine the contextual information to assign the final label to the | Collocation (remote sensing) | Example: Pol-Ice Campaign

### Example: Pol-Ice Campaign  
Map of E-M Bird flights from Pol-Ice campaign along with coincident EMIRAD flights  
Collocations of [sea ice thickness] and [brightness temperature]s taken during the
[Pol-Ice Campaign] are an excellent example since they illustrate many of the most important principles as well as demonstrating the necessity of taking into account the individual case.  The Pol-Ice campaign was conducted in the N. Baltic in March 2007 as part of the SMOS-Ice project in preparation for the launch of the [Soil Moisture and Ocean Salinity satellite].
Because of the low frequency of the SMOS instrument, it is hoped that it will render
information on sea ice thickness, therefore the campaign comprised measurements
of both sea ice thickness and emitted brightness temperature.
Brightness temperatures were measured with the EMIRAD L-band microwave radiometer  
carried on board an airplane. Ice thickness was measured with the E-M Bird ice thickness meter which was carried by a helicopter.  The E-M Bird measures ice thickness with a combination of inductance measurements to determine the location of the ice-water interface and a [LIDAR|laser altimeter] to measure the height of the ice surface.
The map above shows the flight tracks of both instruments which were approximately coincident but obviously subject to pilot error.  
EMIRAD antenna response pattern  
Since the flight paths of both aircraft were approximately linear, the first step in the collocation process was to convert all the coincident flights to Cartesian coordinates with the x-axis being lateral distance and the y-axis transverse distance.  In this way, collocations can be performed in two ways: crudely, by matching only the x distances, and more precisely by matching both coordinates.  
More importantly, the footprint size of the radiometer is many times larger
than that of the E-M Bird meter.  The figure to the left shows
the [radiation pattern|antenna response function] for the radiometer.
The [full width at half maximum] is 31 degrees.  
The figure below illustrates relative measurement locations
from each of the instruments used in the Pol-Ice campaign.
Two overpasses are shown: one from the airplane carrying the
EMIRAD radiometer and one from the helicopter carrying the
E-M Bird instrument.
The x-axis is along the line of the flight path.
EMIRAD footprints are drawn with lines, E-M Bird
inductance measurements are represented by circles
and LIDAR measurements with dots.  
thumb|center|upright=2.5|alt=Pol-Ice campaign measurement locations|Relative measurement locations from P4X to P2A flight track: see above map.
EMIRAD footprints represent the Gaussian standard-deviation, not FWHM.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Overview

## Hyperspectral Imager for the Coastal Ocean  
### Overview  
The Hyperspectral Imager for the Coastal Ocean (HICO) was a hyperspectral earth observation sensor that operated on the International Space Station (ISS) from 2009 to 2014.  HICO collected hyperspectral satellite imagery of the Earth's surface from the ISS.
HICO was a pathfinder or proof-of-concept mission for hyperspectral imaging of the oceans, particularly for optically complex coastal waters. The dataset collected by HICO serves as an example dataset for future hyperspectral satellite missions such as PACE.
HICO was mounted directly on the ISS rather than on a separate unmanned satellite platform (i.e., distinct from the MODIS sensor mounted on Aqua and Terra satellites and from SeaWiFS mounted on OrbView-2 aka Seastar satellite).  As such, HICO was tasked to collect images of certain regions in sync with the daytime orbit path of the ISS. Further, its data record may contain some gaps in time for operational tasks on board the ISS such as spacewalks and docking.  
> See also: Hico (disambiguation)  
Hyperspectral Imager for the Coastal Ocean (HICO) on the International Space Station.  
The Hyperspectral Imager for the Coastal Ocean (HICO) was a hyperspectral earth observation sensor that operated on the [International Space Station|International Space Station (ISS)] from 2009 to 2014.  HICO collected [Hyperspectral imaging|hyperspectral] [satellite imagery] of the Earth's surface from the ISS.  
HICO was a pathfinder or proof-of-concept mission for hyperspectral imaging of the oceans, particularly for optically complex coastal waters. The dataset collected by HICO serves as an example dataset for future hyperspectral satellite missions such as [Plankton, Aerosol, Cloud, ocean Ecosystem|PACE].  
HICO was mounted directly on the ISS rather than on a separate unmanned satellite platform (i.e., distinct from the [MODIS] sensor mounted on Aqua and Terra satellites and from [SeaWiFS] mounted on OrbView-2 aka Seastar satellite).  As such, HICO was tasked to collect images of certain regions in sync with the daytime orbit path of the ISS. Further, its data record may contain some gaps in time for operational tasks on board the ISS such as spacewalks and docking.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | History

### History
HICO was developed by the United States [Office of Naval Research]. The sensor was launched on September 10, 2009, from the [Tanegashima Space Center] in Japan as payload for the ISS on the [H-IIB|H-2B]-304 rocket (including [HTV-1] transfer vehicle). It was installed on September 24, 2009, onto the Japanese Experiment Module Exposed Facility of the [Kibō (ISS module)|Kibo] Laboratory (Japanese Kibo complex) of the ISS by two [Expedition 20|Expedition-20] engineers, ESA astronaut Frank De Winne and NASA astronaut Nicole Stott. HICO was installed concurrently with the RAIDS/Remote Atmospheric and Ionospheric Detection System: together these two systems are referred to as the "HICO and RAIDS Experiment Payload (HREP or HREP-RAIDS)." HICO Collected over 10,000 images during its operating lifetime. and remain freely accessible today.  
HICO stopped collecting data in September 2014 when radiation from a [solar flare] damaged its computer. Attempts to restart the computer were unsuccessful. The last image date and official end of operations was September 13, 2014.  
After the end of its lifetime, HICO and RAIDS Experiment Payload (HREP) was removed from the ISS on August 3, 2018, on the [SpaceX CRS-15] [Dragon spacecraft|Dragon] space capsule after its July–August 2018 resupply mission. The Dragon's trunk section burned up during re-entry, disposing of the HICO instrument and other contents. The flight that offloaded HICO was the fourth ever round-trip cargo flight with a reused Dragon capsule.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products

### Technical specifications and data products
Hyperspectral imagery (right) uses more spectral bands or more colors of light, providing more information to identify different types of terrain.
Hyperspectral satellite sensors(right) detect light radiating from Earth's surface over the full spectrum of visible light, rather than at a few specific "bands" like RGB or multispectral sensors (left and middle).

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Spectral coverage and resolution

#### Spectral coverage and resolution
HICO uses 128 spectral bands from approximately 353&nbsp;nm to 1080&nbsp;nm wavelengths at 5.7&nbsp;nm spectral resolution (band centers 5.7&nbsp;nm apart).

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Spatial coverage and resolution

#### Spatial coverage and resolution
HICO pixels are approximately 90 meters in spatial resolution. Each full scene covers approximately a 42 by 192&nbsp;km rectangle (varying with altitude and angle). High latitude regions of the Earth are not covered. The ISS accomplishes about sixteen 90-minute orbits per day, and the location of the track for orbit moves to the west as Earth rotates. The ISS orbit tracks over the same area on the ground about every three days, including nighttime overpasses.  However, HICO imaging was limited to collect only one scene per orbit, resulting in about seven to eight daylight scenes per day, often spatially scattered throughout the world.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Radiometric resolution

#### Radiometric resolution
HICO data have a [signal-to-noise ratio] of greater than 200-to-1 for water-penetrating wavelengths and assuming 5% albedo. The sensor had high sensitivity in the blue wavelengths and full coverage of water-penetrating wavelengths.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Temporal coverage and resolution

#### Temporal coverage and resolution
HICO collected satellite imagery from September 25, 2009, to September 13, 2014. A maximum of eight daylight scenes were collected per day. In any specific coastal region where scenes were imaged, temporal resolution is patchy. For example, over [Chesapeake Bay] on the United States east coast, 101 scenes were collected over the entire 5-year mission, and 16 scenes were imaged during the calendar year 2012.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Data products

#### Data products
HICO datasets, like other hyperspectral satellite datasets, are large in terms of data volume. For example, one HICO scene requires 120 MB to 700 MB of disk space (depending on format and compression).  Data are available from NASA Ocean Color Web in [Hierarchical Data Format|HDF] file format (similar to [netCDF]).

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Other hyperspectral satellite sensors

#### Other hyperspectral satellite sensors
(partial list)
*Hyperion, launched aboard the [Earth Observing-1|Earth Observing-1 (EO-1)] spacecraft in 2000
*Compact High Resolution Imaging Spectrometer (CHRIS), on [PROBA-1] in 2001
*[SCIAMACHY|Scanning Imaging Absorption Spectrometer for Atmospheric Chartography (SCIAMACHY)] on [ENVISAT] from 2002 to 2012
*PRecursore IperSpettrale della Missione Applicativa (PRISMA), launched 2019 by the Italian Space Agency
*Advanced Hyperspectral Imager (AHSI), onboard China's [Gaofen (satellite)|GaoFen-5] satellite   in 2018
*[Hyperspectral Imaging Satellite|Hyperspectral Imaging Satellite (HySIS)] launched from India in 2018
*HyperScout instruments launched on nanosatellites
*(Planned) Ocean Color Instrument (OCI) on the [Plankton, Aerosol, Cloud, ocean Ecosystem|Plankton, Aerosols, Clouds, and ocean Ecosystems (PACE)] satellite

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Technical specifications and data products | Other Earth science instruments on the ISS

#### Other Earth science instruments on the ISS
(partial list)
*[ISS-RapidScat], which operated from 2014 to 2016.
* Total and Spectral Solar Irradiance Sensor 1 (TSIS-1), which was installed in 2013.
*[SAGE III on ISS|SAGE III], installed in 2017.
*Lightning Imaging Sensor (LIS), installed in 2017.
*[Global Ecosystem Dynamics Investigation|Global Ecosystem Dynamics Investigation (GEDI)] full-waveform LIDAR, installed in 2018.
*[ECOSTRESS|ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS)] instrument, which was delivered to the ISS on the same mission that offloaded HICO in 2018.
*[Orbiting Carbon Observatory 3] (OCO-3), installed in 2019.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | Applications

### Applications
*[Phytoplankton] ecology in general, such as which types of phytoplankton are present in a region of the ocean based on the signature of their pigments and the colors of light those pigments absorb.
*Detection of [Harmful algal bloom|harmful algal blooms (HABs)] by recognizing unique wavelengths of light being emitted by specific types of plankton blooming in large quantities. For example, HICO imagery has been used in the detection of cyanobacteria blooms in inland waters (such as Lake Erie  and Pinto Lake, California  and blooms of the ciliate Mesodinium rubrum in Long Island Sound.
*Mapping [bathymetry] of shallow waters.
*[Colored dissolved organic matter|Dissolved matter] photochemistry in coastal waters.
*[Water quality] monitoring, including variables such as [chlorophyll-a] and suspended particulate matter.
*Maps of terrain, [Vegetation classification|vegetation type], and bottom type.
*Characterization of the [Deepwater Horizon Oil Spill] in April 2010, including collecting images from the area around the explosion sites and from nearby marshes to identify unpolluted water, oil-water mixtures, and emulsified oil strands.

Use the new feature vector and combine the contextual information to assign the final label to the | Hyperspectral Imager for the Coastal Ocean | See also

### See also  
*[Earth observation satellite]
*[Hyperspectral imaging]
*[Imaging spectroscopy]
*[International Space Station]
*[Kibo (ISS module)]
*[Ocean color]
*[Plankton, Aerosol, Cloud, ocean Ecosystem]
*[Scientific research on the International Space Station]

Use the new feature vector and combine the contextual information to assign the final label to the | Binary image | Usages | 1-bit pixel art

### Usages  
#### 1-bit pixel art
1-bit pixel art combining imagery with text
Binary [pixelart], better known as 1-Bit or 1bit art, has been a method of displaying graphics since early computers. While
early computers such as the [ZX81] used the restriction as a necessity of the hardware, hand-held LCD games such as [Game & Watch] and [Tamagotchi], alongside early computers with a focus on graphic user interfaces like the [Macintosh_128K|Macintosh] made large steps in promoting the culture, technique and aesthetic of the restrictions of 1-bit art.  
Modern examples of 1bit art are visible in indie videogames and other digital art. Best-seller games like [Gato Roboto], [Return of the Obra Dinn], [Minit] and [World of Horror] use 1bit as a style to give their games a retro feel or to simply save the graphic designers time in development. There are also game consoles focused on 1-Bit graphics, such as the handheld [Playdate (console)|Playdate] console, which is also used in education in favor of more involved engines.  
For pixel artists, 1-Bit has become a common challenge for creating art. The pixelart contest Pixtogether required its participants to use only two colours for its 10th monthly contest. Not a lot of artists mainly do 1bit art, but many of them stay in contact with each other to exchange knowledge about working with the restriction, and hosting own collaborations.  
Brandon James Greer, who makes popular YouTube tutorials on 1bit and other pixel artwork, says that "the restriction leads to some unique approaches" and that working in 1-Bit is "a very fun and unique challenge".  
While 1bit can be called an [art style] itself, each piece falls under a second style too. Obvious differences in 1bit art styles are for example whether, how much and what kind of [dithering] is being used, the [image resolution], the use of [Outline drawing|outlines] and how detailed the artwork is.

Use the new feature vector and combine the contextual information to assign the final label to the | Binary image | Usages | Oversampled binary image sensors

#### Oversampled binary image sensors
An [oversampled binary image sensor] is a digital [image sensor] reminiscent of traditional photographic film. Each pixel in the sensor has a binary response, giving only a one-bit quantized measurement of the local light intensity.

Use the new feature vector and combine the contextual information to assign the final label to the | Radar remote sensing | Overview

## Radar remote sensing  
### Overview  
Radar remote sensing is a type of active remote Sensing which uses electromagnetic energy backscattered from ground targets to extract physical and dielectric behavior. It is different from passive remote sensing, the most common type, as the electromagnetic radiation (EMR) is produced by the emitters and they transmit radiation at radio wavelengths (i.e. from around 1 cm to several meters) and sensors use the measured return to infer properties of the Earth's surface. Radar remote sensing uses long-wavelength energy that penetrates through clouds and is sensitive to changes in vegetation physical structure. Thus, it has advantage in its capability of all-hour and all-weather imaging.
Its capability of all-weather imaging and specific range of EMR spectrum enables it to be applicable in Digital elevation mapping, Vegetation cover mapping, Soil mapping, Archeological applications etc. Various satellite based sensors are using this kind of technology to produce Radar based remote sensing data (see RADARSAT, TerraSAR-X, Magellan).  
Synthetic aperture radar image of [Death Valley] colored using [polarimetry]  
Radar remote sensing is a type of [Remote Sensing|active remote Sensing] which uses electromagnetic energy backscattered from ground targets to extract physical and dielectric behavior. It is different from passive remote sensing, the most common type, as the electromagnetic radiation (EMR) is produced by the emitters and they transmit radiation at radio wavelengths (i.e. from around 1&nbsp;cm to several meters) and sensors use the measured return to infer properties of the Earth's surface. Radar remote sensing uses long-wavelength energy that penetrates through clouds and is sensitive to changes in vegetation physical structure. Thus, it has advantage in its capability of all-hour and all-weather imaging.  
Its capability of all-weather imaging and specific range of EMR spectrum enables it to be applicable in Digital elevation mapping, Vegetation cover mapping, Soil mapping, Archeological applications etc. Various satellite based sensors are using this kind of technology to produce Radar based remote sensing data (see [RADARSAT], [TerraSAR-X], [Magellan probe|Magellan]).

Use the new feature vector and combine the contextual information to assign the final label to the | Landsat program | Spatial and spectral resolution

### Spatial and spectral resolution
Landsat 1 through 5 carried the Landsat [Multispectral Scanner] (MSS). Landsat 4 and 5 carried both the MSS and [Thematic Mapper] (TM) instruments. Landsat 7 uses the [Landsat 7|Enhanced Thematic Mapper Plus] (ETM+) scanner. Landsat 8 uses two instruments, the [Operational Land Imager] (OLI) for [Free-space optical communication|optical bands] and the Thermal Infrared Sensor (TIRS) for [Thermal remote sensing|thermal bands]. The band designations, [Passband|bandpasses], and [pixel] sizes for the Landsat instruments are:
{| class="wikitable"  
!Landsat 1–3 MSS
!Landsat 4–5 MSS
!Wavelength (micrometers)
!Resolution (meters)  
* Original MSS pixel size was 79 x 57 meters; production systems now resample the data to 60 meters.
{| class="wikitable"  
!Bands
!Wavelength (micrometers)
!Resolution (meters)  
* TM Band 6 was acquired at 120-meter resolution, but products are resampled to 30-meter pixels.
{| class="wikitable"  
!Bands
!Wavelength (micrometers)
!Resolution (meters)  
* ETM+ Band 6 is acquired at 60-meter resolution, but products are [image scaling|resampled] to 30-meter pixels.  
The spectral band placement for each sensor of Landsat  
{| class="wikitable"  
!Bands
!Wavelength (micrometers)
!Resolution (meters)  
* TIRS bands are acquired at 100 meter resolution, but are resampled to 30 meter resolution in the delivered data product.  
An advantage of Landsat imagery, and [remote sensing] in general, is that it provides data at a [Synoptic scale meteorology|synoptic] global level that is impossible to replicate with in situ measurements. However, there are tradeoffs between the local detail of the measurements ([Radiometry|radiometric] resolution, number of [spectral band]s) and the spatial scale of the measured area. Landsat imagery is coarse in [spatial resolution] compared to other remote sensing methods, such as imagery from airplanes. Landsat's spatial resolution is relatively high compared to other satellites, yet its [Satellite revisit period|revisit time] is relatively less frequent.

Use the new feature vector and combine the contextual information to assign the final label to the | Landsat program | MultiSpectral Scanner (MSS)

### MultiSpectral Scanner (MSS)
The Landsat program incorporated the [Multispectral Scanner] (MSS) from its first mission to its fifth. The MSS gave the United States an advantage in satellite imaging, facilitating the launch of Landsat ahead of the French [SPOT (satellite)|SPOT satellite].  
The MSS was unique in its design. It employed a moving mirror, rather than a static [camera], capturing Earth's images in four distinct spectral bands. This capability allowed the MSS to record variations in [Earthlight|sunlight reflected from the Earth]. Notably, Landsat 3's MSS was further advanced, with an added capability to detect [Thermal radiation|heat radiation].  
One of the prominent features of the MSS was its consistent imaging. Each captured frame represented an area on the Earth's surface approximately 83 meters in length and 68 meters in width. Additionally, the system was designed to ensure a continuous image sweep across a swath equivalent to 185&nbsp;km on the Earth's surface. The MSS's design also emphasized precision; by precisely timing the mirror's movements, it ensured that consecutive images did not overlap.

Use the new feature vector and combine the contextual information to assign the final label to the | Landsat program | Uses of Landsat imagery

### Uses of Landsat imagery
One year after launch, Landsat 8 imagery had over one million file downloads by data users.
Landsat data provides information that allows scientists to predict [species distribution] and detect both naturally occurring and [Human impact on the environment|human-generated changes] over a greater scale than traditional data from field work. The different spectral bands used on satellites in the Landsat program provide many applications, ranging from [ecology] to geopolitical matters. [Land cover]-determination is a common use of Landsat imagery around the world.  
Landsat imagery provides one of the longest uninterrupted time series available from any single remote sensing program, spanning from 1972 to the present. Looking to the future, the successful launch of [Landsat 9] in 2021 shows that this time series will continue.  
irrigated fields near [Garden City, Kansas], taken by the [Landsat 7] satellite]]
In 2015, the Landsat Advisory Group of the National Geospatial Advisory Committee reported that the top 16 applications of Landsat imagery produced savings of approximately 350 million to over 436 million dollars each year for federal and state governments, non-governmental organizations ([Non-governmental organization|NGOs]), and the private sector. That estimate did not include further savings from other uses beyond the top sixteen categories. The top 16 categories for Landsat imagery use, listed in order of estimated annual savings for users, are:

[Wildfire] risk assessment | Forestry

# [Wildfire] risk assessment  
##### Forestry
An ecological study used 16 [Orthophoto|ortho-rectified] Landsat images to generate a land cover map of [Mozambique]'s [mangrove] forest. The main objective was to measure the mangrove cover and above-ground [biomass] in this zone, that until now could only be estimated. The cover was found to have a 93% accuracy of 2909 square kilometers (27% lower than previous estimates). Additionally, the study helped confirm that geological setting has a greater influence on biomass distribution than latitude - the mangrove area is spread across 16° of latitude but the biomass volume of it was affected more strongly by geographic conditions.

[Wildfire] risk assessment | Climate change and environmental disasters | Urban development

##### Urban development
Landsat imagery gives a [Time-lapse photography|time-lapse]-like series of images of development. [Human development (economics)|Human development], specifically, can be measured by the size a city grows over time. Further than just [Census|population estimates] and [energy consumption], Landsat imagery gives an insight into the type of [Urban planning|urban development], and studies aspects of social and political change through visible change. In [Beijing] for example, a series of [Ring roads of Beijing|ring roads] started to develop in the 1980s following the economic reform of 1970, and the change in development rate and construction rate was accelerated during this time period.

[Wildfire] risk assessment | Recent and future Landsat satellites

### Recent and future Landsat satellites
Landsat 8/9 and Landsat Next spectral band comparison  
[Landsat 8] launched on 11 February 2013. It was launched on an [Atlas V] 401 from [Vandenberg Air Force Base] by the [Launch Services Program]. It will continue to obtain valuable data and imagery to be used in agriculture, education, business, science, and government. The new satellite was assembled in [Arizona] by [Orbital Sciences Corporation].  
[Landsat 9] launched on September 27, 2021. During FY2014 financial planning "appropriators chided NASA for unrealistic expectations that a Landsat 9 would cost US$1 billion, and capped spending at US$650 million" according to a report by the [Congressional Research Service]. [United States Senate] appropriators advised NASA to plan for a launch no later than 2020. Funding for the development of a low-cost thermal infrared (TIR) free-flying satellite for launch in 2019 was also proposed to ensure data continuity by flying in formation with Landsat 8.  
Landsat Next is planned to launch in late 2030/early 2031 and will measure 26 spectral bands; current Landsat's 8 and 9 measure 11 each.

[Wildfire] risk assessment | Mapping Services Agreement | Overview

## Mapping Services Agreement  
### Overview  
The Mapping Services Agreement (MSA) is a licensing contract between local authorities in the United Kingdom and suppliers of geographic data. Most of its contents are covered by commercial in confidence requirements. The general outcome of the MSA, however, is the supply of geographic data to local authorities and the defining of licensing issues regarding address data.
The MSA replaced an existing agreement for the supply of geographic data between local authorities, police, fire and other emergency services, and the national mapping agency, Ordnance Survey (OS). It also complied with European Union rules on procurement. As a result, suppliers of geographic information had to go through a process of open procurement managed on behalf of local authorities by the IDeA, an umbrella government organisation coordinating and promoting local authority good practice.
The result was that three suppliers were selected: Ordnance Survey, Intermap and Intelligent Addressing. For local authorities this meant little change in the supply of mapping data which continued to be supplied by OS. Intermap, however, supplied height data not previously covered by other arrangements.
Perhaps the most significant outcome of the MSA is the resolution of ownership, licensing and royalty issues that had existed between local authorities and OS. In the agreement the address databases (called Local Land and Property Gazetteers – LLPG) maintained by local authorities acknowledged the partial input of OS's address product, Address Point to these databases. In effect local authorities became "value added resellers" of ADDRESS-POINT and are required to pay OS royalties for the proportion of their use of the OS product.
The agreement also puts into place funding for the National Land and Property Gazetteer (NLPG), the UK's national address infrastructure, which is made up of the LLPGs compiled by local authorities. It also enables the NLPG to compel local authorities to maintain their LLPGs and thus ensure the nationwide coverage and compliance by local authorities.
The Mapping Services Agreement contract expired on 31 March 2012 and was replaced by the Data Co-operation Agreement.  
The Mapping Services Agreement (MSA) is a licensing [contract] between [local authorities] in the [United Kingdom] and suppliers of [geographic data]. Most of its contents are covered by commercial in confidence requirements. The general outcome of the MSA, however, is the supply of geographic data to local authorities and the defining of licensing issues regarding address data.  
The MSA replaced an existing agreement for the supply of geographic data between local authorities, police, fire and other emergency services, and the national mapping agency, [Ordnance Survey] (OS). It also complied with [European Union] rules on procurement. As a result, suppliers of geographic information had to go through a process of open procurement managed on behalf of local authorities by the IDeA, an umbrella government organisation coordinating and promoting local authority good practice.  
The result was that three suppliers were selected: Ordnance Survey, Intermap and Intelligent Addressing. For local authorities this meant little change in the supply of mapping data which continued to be supplied by OS. Intermap, however, supplied height data not previously covered by other arrangements.  
Perhaps the most significant outcome of the MSA is the resolution of ownership, licensing and royalty issues that had existed between local authorities and OS. In the agreement the address databases (called [Local Land and Property Gazetteer]s – LLPG) maintained by local authorities acknowledged the partial input of OS's address product, [Address Point] to these databases. In effect local authorities became "value added resellers" of ADDRESS-POINT and are required to pay OS royalties for the proportion of their use of the OS product.  
The agreement also puts into place funding for the [National Land and Property Gazetteer] (NLPG), the UK's national address infrastructure, which is made up of the LLPGs compiled by local authorities. It also enables the NLPG to compel local authorities to maintain their LLPGs and thus ensure the nationwide coverage and compliance by local authorities.  
The Mapping Services Agreement contract expired on 31 March 2012 and was replaced by the Data Co-operation Agreement.

[Wildfire] risk assessment | Asian Association on Remote Sensing | Overview

## Asian Association on Remote Sensing  
### Overview  
Asian Association on Remote Sensing (AARS) is a non-governmental organization established in 1981 to promote remote sensing in the Asia-Pacific region; it currently has members from 29 countries.  
Asian Association on Remote Sensing (AARS) is a non-governmental organization established in 1981 to promote [remote sensing] in the [Asia-Pacific region]; it currently has members from 29 countries.

[Wildfire] risk assessment | Asian Association on Remote Sensing | Members

### Members
Its members include:
* [Indian Society of Remote Sensing]
* [Surveying & Spatial Sciences Institute]
* [Malaysian Remote Sensing Agency]
* [Japan Society of Photogrammetry and Remote Sensing]
* [SPARRSO]
* [Institute of Remote Sensing and Digital Earth]
* [Korean Society of Remote Sensing]

[Wildfire] risk assessment | Space-based radar | Overview

## Space-based radar  
### Overview  
Space-based radar or spaceborne radar is a radar operating in outer space;
orbiting radar is a radar in orbit and
Earth orbiting radar is a radar in geocentric orbit.
A number of Earth-observing satellites, such as RADARSAT, have employed synthetic aperture radar (SAR) to obtain terrain and land-cover information about the Earth.  
ORS-2
Space-based radar or spaceborne radar is a [radar] operating in [outer space];
orbiting radar is a radar in [orbit] and
Earth orbiting radar is a radar in [geocentric orbit].
A number of [Earth-observing satellite]s, such as [RADARSAT], have employed [synthetic aperture radar] (SAR) to obtain terrain and land-cover information about the [Earth].

[Wildfire] risk assessment | Space-based radar | Planetary radars

### Planetary radars
Most of the radars flown as payload in planetary missions (i.e., not considering avionics radar, such as docking and landing radars used in [Project Apollo|Apollo] and [Apollo Lunar Module|LEM]) belong to two categories: imaging radars and sounders.  
Imaging radars: [Synthetic aperture radar]s are the only instruments capable of penetrating heavy cloud cover around planets such as [Venus], which was the first target for such missions. Two Soviet spacecraft ([Venera 15] and [Venera 16]) imaged the planet in 1983 and 1984 using SAR and [Radar altimeter]s. The [Magellan probe] also imaged Venus in 1990 and 1994.  
The only other target of an [imaging radar] mission has been [Titan (moon)|Titan], the largest moon of [Saturn], in order to penetrate its opaque atmosphere. The radar of the [Cassini probe|Cassini probe], which orbited [Saturn] between 2004 and 2017, provided images of [Titan (moon)|Titan's] surface during each fly-by of the moon. The Cassini radar was a multimode system and could operate as [Synthetic Aperture Radar], [radar altimeter], [scatterometer] and [radiometer].  
Sounding radars: these are low-frequency (normally, HF - 3 to 30&nbsp;MHz - or lower) ground-penetrating [Radars], used to acquire data about the planet sub-surface structure. Their low operating frequency allow them to penetrate hundreds of meters, or even kilometers, below the surface. Synthetic aperture techniques are normally exploited to reduce the ground footprint (due to the low operating frequency and the small allowable [Antenna (radio)|antenna] dimensions, the beam is very wide) and, thus, the unwanted echo from other surface objects.  
The first radar sounder flown was [ALSE] (Apollo Lunar Sounder Experiment) on board [Apollo 17] in 1972.  
Other sounder instruments flown (in this case around [Mars]), are [MARSIS] (Mars Advanced Radar for SubSurface and [Ionosphere] Sounding) on board the [European Space Agency]'s [Mars Express] probe, and [SHARAD] (mars SHAllow RADar sounder) on [Jet Propulsion Laboratory|JPL]'s [Mars Reconnaissance Orbiter] (MRO). Both are currently operational. A radar sounder is also used on the Japanese Moon probe [SELENE], launched September 14, 2007.  
A similar instrument (primarily devoted to ionospheric [Plasma (physics)|plasma] probing) was embarked on the Japanese Martian mission [Nozomi (probe)|Nozomi] (launched in 1998 but lost).

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Overview

## Committee on Earth Observation Satellites  
### Overview  
The Committee on Earth Observation Satellites (CEOS) is an international coordination body established in 1984 to facilitate cooperation among space agencies operating Earth observation satellite systems.
CEOS consists of 35 member organisations (operating or planning Earth observation satellites) alongside 32 associate organisations. Membership across both categories spans over 35 countries, as well as numerous intergovernmental organisations.
While membership is restricted to national or international organisations; non-members may participate in thematic and technical activities. Commercial, academic and non-governmental organisations participate in Working Groups and Virtual Constellations.  
The Committee on Earth Observation Satellites (CEOS) is an international coordination body established in 1984 to facilitate cooperation among [Space agency|space agencies] operating [Earth observation satellites|Earth observation satellite systems].
CEOS consists of 35 member organisations (operating or planning Earth observation satellites) alongside 32 associate organisations. Membership across both categories spans over 35 countries, as well as numerous intergovernmental organisations.
While membership is restricted to national or international organisations; non-members may participate in thematic and technical activities. Commercial, academic and non-governmental organisations participate in Working Groups and Virtual Constellations.

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Background

### Background
CEOS was established in September 1984 in response to a recommendation from a Panel of Experts on Remote Sensing from Space and set up under the aegis of the G7 Economic Summit of Industrial Nations Working Group on Growth, Technology, and Employment. This Panel recognized the multidisciplinary nature of space-based Earth observations and the value of coordinating international Earth observation efforts to benefit society.

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Mission

### Mission
The stated mission of CEOS is to coordinate civil space-based Earth observation programmes internationally and to promote the exchange of data in support of societal and environmental decision-making.  
In October 2024, during its 40th anniversary plenary in Montreal, CEOS adopted the Montreal Statement, reaffirming organisational priorities and reflecting on four decades of activity.

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Organisational Structure

### Organisational Structure
Organisational Chart for the Committee on Earth Observation Satellites (CEOS)
CEOS leadership roles are supported by members on a rotating basis.

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Organisational Structure | CEOS Chair and CEOS Plenary

#### CEOS Chair and CEOS Plenary
The CEOS Chair is a senior space agency official that serves a one-year term, and hosts the annual CEOS Plenary.

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Organisational Structure | Strategic Implementation Team (SIT)

#### Strategic Implementation Team (SIT)
The Strategic Implementation Team (SIT) is formed of all CEOS Members and Associates, and focuses on strategic guidance with regard to governance, stakeholders, and the accomplishment of the CEOS Work Plan. The SIT is led by the SIT Chair, who serves a two-year term, and hosts the annual SIT meeting and SIT Technical Workshop.  
The five CEOS Working Groups are:  
* Working Group on Capacity Building & Data Democracy (WGCapD)
* CEOS/CGMS Joint Working Group on Climate (WGClimate)
* Working Group on Calibration & Validation (WGCV)
* Working Group on Disasters (WGDisasters)
* Working Group on Information Systems & Services (WGISS)

[Wildfire] risk assessment | Committee on Earth Observation Satellites | Organisational Structure | Virtual Constellations

#### Virtual Constellations
A CEOS Virtual Constellation is a set of space and [ground segment] capabilities operating together in a coordinated manner, in effect a virtual system that overlaps in coverage in order to meet a combined and common set of Earth observation requirements.  
The eight CEOS Virtual Constellations are:  
* Atmospheric Composition (AC-VC)
* Coastal Observations Applications Services and Tools (COAST-VC)
* Land Surface Imaging (LSI-VC)
* Ocean Colour Radiometry (OCR-VC)
* Ocean Surface Topography (OST-VC)
* Ocean Surface Vector Wind (OSVW-VC)
* Precipitation (P-VC)
* [Sea surface temperature|Sea Surface Temperature] (SST-VC)

[Wildfire] risk assessment | Remote sensing | Overview

## Remote sensing  
### Overview  
Remote sensing is the acquisition of information about an object or phenomenon without making physical contact with the object, in contrast to in situ or on-site observation. The term is applied especially to acquiring information about Earth and other planets. Remote sensing is used in numerous fields, including geophysics, geography, land surveying and most Earth science disciplines (e.g. exploration geophysics, hydrology, ecology, meteorology, oceanography, glaciology, geology). It also has military, intelligence, commercial, economic, planning, and humanitarian applications, among others.
In current usage, the term remote sensing generally refers to the use of satellite- or airborne-based sensor technologies to detect and classify objects on Earth. It includes the surface and the atmosphere and oceans, based on propagated signals (e.g. electromagnetic radiation). It may be split into "active" remote sensing (when a signal is emitted by a sensor mounted on a satellite or aircraft to the object and its reflection is detected by the sensor) and "passive" remote sensing (when the reflection of sunlight is detected by the sensor).  
Synthetic aperture radar image of [Death Valley] colored using [polarimetry]  
Remote sensing is the acquisition of [information] about an [physical object|object] or [phenomenon] without making physical contact with the object, in contrast to [in situ] or on-site [observation]. The term is applied especially to acquiring information about [Earth] and other [planet]s. Remote sensing is used in numerous fields, including [geophysics], [geography], land [surveying] and most [Earth science] disciplines (e.g. [exploration geophysics], [hydrology], [ecology], [meteorology], [oceanography], [glaciology], [geology]). It also has military, intelligence, commercial, economic, planning, and humanitarian applications, among others.  
In current usage, the term remote sensing generally refers to the use of [satellite]- or airborne-based [sensor] technologies to detect and classify objects on Earth. It includes the surface and the [atmosphere] and [oceans], based on [wave propagation|propagated signals] (e.g. [electromagnetic radiation]). It may be split into "active" remote sensing (when a signal is emitted by a sensor mounted on a satellite or aircraft to the object and its reflection is detected by the sensor) and "passive" remote sensing (when the reflection of sunlight is detected by the sensor).  
### Overview
This video is about how Landsat was used to identify areas of conservation in the [Democratic Republic of the Congo], and how it was used to help map an area called [Maringa-Lopori-Wamba Landscape|MLW] in the north.
Remote sensing can be divided into two types of methods: Passive remote sensing and active remote sensing. Passive sensors gather radiation that is emitted or reflected by the object or surrounding areas. Reflected [sunlight] is the most common source of radiation measured by passive sensors. Examples of passive remote sensors include film [photography], [infrared], [charge-coupled devices], and [radiometers]. Active collection, on the other hand, emits energy in order to scan objects and areas whereupon a sensor then detects and measures the radiation that is reflected or backscattered from the target. [Radar] and [lidar] are examples of active remote sensing where the time delay between emission and return is measured, establishing the location, speed and direction of an object.
Illustration of remote sensing  
Remote sensing makes it possible to collect data of dangerous or inaccessible areas. Remote sensing applications include monitoring [deforestation] in areas such as the [Amazon Basin], [glacier|glacial] features in Arctic and Antarctic regions, and [depth sounding] of coastal and ocean depths. Military collection during the [Cold War] made use of stand-off collection of data about dangerous border areas. Remote sensing also replaces costly and slow data collection on the ground, ensuring in the process that areas or objects are not disturbed.  
Orbital platforms collect and transmit data from different parts of the [electromagnetic spectrum], which in conjunction with larger scale aerial or ground-based sensing and analysis, provides researchers with enough information to monitor trends such as [El Niño] and other natural long and short term phenomena. Other uses include different areas of the [earth science]s such as [natural resource management], agricultural fields such as land usage and conservation, [greenhouse gas monitoring], oil spill detection and monitoring, and national security and overhead, ground-based and stand-off collection on border areas.

[Wildfire] risk assessment | Remote sensing | Data acquisition

### Data acquisition
The basis for multispectral collection and analysis is that of examined areas or objects that reflect or emit radiation that stand out from surrounding areas. For a summary of major remote sensing satellite systems see the overview table.

[Wildfire] risk assessment | Remote sensing | Data acquisition | Applications of remote sensing

#### Applications of remote sensing
> Further: Remote sensing (geology)
> Further: Remote sensing in archaeology
> See also: Mobile radar observation of tornadoes  
Aswan Dam, Egypt taken by Umbra]]Conventional radar is mostly associated with [air traffic control], early warning, and certain large-scale meteorological data. [Doppler radar] is used by local law enforcements' monitoring of speed limits and in enhanced [Weather radar|meteorological collection] such as wind speed and direction within weather systems in addition to precipitation location and intensity. Other types of active collection includes [plasma (physics)|plasmas] in the [ionosphere]. [Interferometric synthetic aperture radar] is used to produce precise [digital elevation model]s of large scale terrain (See [RADARSAT], [TerraSAR-X], [Magellan probe|Magellan]). Laser and [radar altimeter|radar] [altimeter]s on satellites have provided a wide range of data. By measuring the bulges of water caused by gravity, they map features on the seafloor to a resolution of a mile or so. By measuring the height and wavelength of ocean waves, the altimeters measure wind speeds and direction, and surface ocean currents and directions. Ultrasound (acoustic) and radar tide gauges are used to measure sea level, tides and wave direction in coastal and offshore tide gauges.  
[Light detection and ranging] (LiDAR) is used for weapon ranging, laser illuminated homing of projectiles, and to detect and measure the concentration of various chemicals in the atmosphere while airborne LiDAR can be used to measure the heights of objects and features on the ground more accurately than radar technology. LiDAR can be used to detect ground surface changes typically by creating Digital Surface Models (DSMs) or Digital Elevation Models (DEMs). Vegetation remote sensing is a principal application of LIDAR.  
The most common instruments in use are [Radiometer|radiometers] and [photometer]s, which collect reflected and emitted radiation in a wide range of frequencies. The most prevalent of these frequencies are visible and infrared sensors, followed by microwave, gamma-ray, and rarely, ultraviolet. They may also be used to detect the [emission spectra] of various chemicals, providing data on chemical concentrations in the atmosphere. Radiometers are also used at night, as [Light pollution|artificial light emissions] are a key signature of human activity. Applications include remote sensing of population, GDP, and damage to infrastructure from war or disasters. Radiometers and radar onboard of satellites can be also used to monitor volcanic eruptions. [Polarimetry#Imaging|Spectropolarimetric Imaging] has been reported to be useful for target tracking purposes by researchers at the [United States Army Research Laboratory|U.S. Army Research Laboratory]. They determined that manmade items possess polarimetric signatures that are not found in natural objects. These conclusions were drawn from the imaging of military trucks, like the [Humvee], and trailers with their [Acousto-optic modulator|acousto-optic tunable filter] dual [hyperspectral] and spectropolarimetric VNIR Spectropolarimetric Imager.  
[stereoscopy|Stereographic pairs] of [aerial photograph]s have often been used to make [topographic map]s by imagery and terrain analysts in trafficability and highway departments for potential routes, in addition to modelling terrestrial habitat features.  
Simultaneous multi-spectral platforms such as Landsat have been in use since the early 1970s. These thematic mappers take images in multiple wavelengths and are usually found on [Earth observation satellite]s, including (for example) the [Landsat program] or the [IKONOS] satellite. Maps of [land cover] and [land use] from thematic mapping can be used to prospect for minerals, detect or monitor land usage, detect invasive vegetation, deforestation, and examine the health of indigenous plants and crops ([satellite crop monitoring]), including entire farming regions or forests. Prominent scientists using remote sensing for this purpose include [Janet Franklin] and [Ruth DeFries]. Landsat images are used by regulatory agencies such as KYDOW to indicate water quality parameters including Secchi depth, chlorophyll density, and total phosphorus content. [Weather satellite]s are used in meteorology and climatology.  
[Hyperspectral imaging] produces image cubes where each pixel has full spectral information with imaging narrow spectral bands over a contiguous spectral range. Hyperspectral imagers are used in various applications including mineralogy, biology, defence, and environmental measurements. Within the scope of the combat against [desertification], remote sensing allows researchers to follow up and monitor risk areas in the long term, to determine desertification factors, to support decision-makers in defining relevant measures of environmental management, and to assess their impacts. Remotely sensed multi- and hyperspectral images can be used for assessing biodiversity at different spatial scales. Since the spectral properties of different plants species are unique, it is possible to get information about properties that relates to biodiversity such as habitat heterogeneity, spectral diversity and plant functional trait. Remote sensing has also been used to detect rare plants to aid in conservation efforts. Prediction, detection, and the ability to record biophysical conditions were possible from medium to very high resolutions. Remote sensing is often utilized in the collection of agricultural and environmental statistics, usually combining classified satellite images with [ground truth] data collected on a sample selected on an [area sampling frame]

[Wildfire] risk assessment | Remote sensing | Data acquisition | Geodetic

#### Geodetic
> Further: Satellite geodesy
[Geodesy|Geodetic] remote sensing can be [gravimetry|gravimetric] or geometric. Overhead gravity data collection was first used in aerial submarine detection. This data revealed minute perturbations in the Earth's [gravitational field] that may be used to determine changes in the mass distribution of the Earth, which in turn may be used for geophysical studies, as in [GRACE (satellite)|GRACE]. Geometric remote sensing includes position and deformation [Radar imaging|imaging] using [InSAR], LIDAR, etc.

[Wildfire] risk assessment | Remote sensing | Data acquisition | Acoustic and near-acoustic

#### Acoustic and near-acoustic  
Three main types of acoustic and near-acoustic remote sensing exist: [Sonar] – passive sonar, listening for the sound made by another object (a vessel, a whale etc.); active sonar, emitting pulses of sounds and listening for echoes, used for detecting, ranging and measurements of underwater objects and terrain. [seismograph|Seismograms] taken at different locations can locate and measure [earthquake]s (after they occur) by comparing the relative intensity and precise timings. [Ultrasound] acoustic sensing is made up of ultrasound sensors that emit high-frequency pulses and listening for echoes, used for detecting water waves and water level, as in tide gauges or for towing tanks.  
To coordinate a series of large-scale observations, most sensing systems depend on the following: platform location and the orientation of the sensor. High-end instruments now often use positional information from [satellite navigation system]s. The rotation and orientation are often provided within a degree or two with electronic compasses. Compasses can measure not just azimuth (i. e. degrees to magnetic north), but also altitude (degrees above the horizon), since the magnetic field curves into the Earth at different angles at different latitudes. More exact orientations require [inertial navigation system|gyroscopic-aided orientation], periodically realigned by different methods including navigation from stars or known benchmarks.

[Wildfire] risk assessment | Remote sensing | Data acquisition | Gamma rays

#### Gamma rays
There are applications of [gamma rays] to mineral exploration through remote sensing. In 1972 more than $2 million  were spent on remote sensing applications with gamma rays to mineral exploration. Gamma rays are used to search for deposits of [uranium]. By observing radioactivity from potassium, [porphyry copper] deposits can be located. A high ratio of uranium to thorium has been found to be related to the presence of hydrothermal copper deposits. Radiation patterns have also been known to occur above oil and gas fields, but some of these patterns were thought to be due to surface soils instead of oil and gas.

[Wildfire] risk assessment | Remote sensing | Data characteristics

### Data characteristics  
The quality of remote sensing data consists of its spatial, spectral, radiometric and temporal resolutions.  
; [Spatial resolution]: The size of a [pixel] that is recorded in a [raster graphics|raster image] – typically pixels may correspond to square areas ranging in side length from .
; [Spectral resolution]: The bandwidth of the different frequency bands recorded – usually, this is related to the number of frequency bands recorded by the platform. Current [Landsat] collection is that of seven bands, including several in the [infrared] spectrum, ranging from a spectral resolution of 0.7 to 2.1 μm. The Hyperion sensor on Earth Observing-1 resolves 220 bands from 0.4 to 2.5 μm, with a spectral resolution of 0.10 to 0.11 μm per band.
; [Radiometric resolution]: The number of different intensities of radiation the sensor is able to distinguish. Typically, this ranges from 8 to 14 bits, corresponding to 256 levels of the gray scale and up to 16,384 intensities or "shades" of colour, in each band. It also depends on the instrument [noise].
; [Temporal resolution]: The frequency of flyovers by the satellite or plane, and is only relevant in time-series studies or those requiring an averaged or mosaic image as in deforesting monitoring. This was first used by the intelligence community where repeated coverage revealed changes in infrastructure, the deployment of units or the modification/introduction of equipment. Cloud cover over a given area or object makes it necessary to repeat the collection of said location.

[Wildfire] risk assessment | Remote sensing | Data processing

### Data processing
In order to create sensor-based maps, most remote sensing systems expect to extrapolate sensor data in relation to a reference point including distances between known points on the ground. This depends on the type of sensor used. For example, in conventional photographs, distances are accurate in the center of the image, with the distortion of measurements increasing the farther you get from the center. Another factor is that of the platen against which the film is pressed can cause severe errors when photographs are used to measure ground distances. The step in which this problem is resolved is called [georeference|georeferencing] and involves computer-aided matching of points in the image (typically 30 or more points per image) which is extrapolated with the use of an established benchmark, "warping" the image to produce accurate spatial data. As of the early 1990s, most satellite images are sold fully georeferenced.  
In addition, images may need to be radiometrically and atmospherically corrected.  
; Radiometric correction: Allows avoidance of radiometric errors and distortions. The illumination of objects on the Earth's surface is uneven because of different properties of the relief. This factor is taken into account in the method of radiometric distortion correction. Radiometric correction gives a scale to the pixel values, e. g. the monochromatic scale of 0 to 255 will be converted to actual radiance values.
; Topographic correction (also called terrain correction): In rugged mountains, as a result of terrain, the effective illumination of pixels varies considerably. In a remote sensing image, the pixel on the shady slope receives weak illumination and has a low radiance value, in contrast, the pixel on the sunny slope receives strong illumination and has a high radiance value. For the same object, the pixel radiance value on the shady slope will be different from that on the sunny slope. Additionally, different objects may have similar radiance values. These ambiguities seriously affected remote sensing image information extraction accuracy in mountainous areas. It became the main obstacle to the further application of remote sensing images. The purpose of topographic correction is to eliminate this effect, recovering the true reflectivity or radiance of objects in horizontal conditions. It is the premise of [quantitative remote sensing] application.
; [Atmospheric correction]: Elimination of atmospheric haze by rescaling each frequency band so that its minimum value (usually realised in water bodies) corresponds to a pixel value of 0. The digitizing of data also makes it possible to manipulate the data by changing gray-scale values.  
Interpretation is the critical process of making sense of the data. The first application was that of aerial photographic collection which used the following process; spatial measurement through the use of a [light table] in both conventional single or stereographic coverage, added skills such as the use of photogrammetry, the use of photomosaics, repeat coverage, Making use of objects' known dimensions in order to detect modifications. Image Analysis is the recently developed automated computer-aided application that is in increasing use.  
[Object-Based Image Analysis] (OBIA) is a sub-discipline of GI-Science devoted to partitioning remote sensing (RS) imagery into meaningful image-objects, and assessing their characteristics through spatial, spectral and temporal scale.  
Old data from remote sensing is often valuable because it may provide the only long-term data for a large extent of geography. At the same time, the data is often complex to interpret, and bulky to store. Modern systems tend to store the data digitally, often with [lossless compression]. The difficulty with this approach is that the data is fragile, the format may be archaic, and the data may be easy to falsify. One of the best systems for archiving data series is as computer-generated machine-readable [ultrafiche], usually in typefonts such as [OCR-B], or as digitized half-tone images. Ultrafiches survive well in standard libraries, with lifetimes of several centuries. They can be created, copied, filed and retrieved by automated systems. They are about as compact as archival magnetic media, and yet can be read by human beings with minimal, standardized equipment.  
Generally speaking, remote sensing works on the principle of the [inverse problem]: while the object or phenomenon of interest (the state) may not be directly measured, there exists some other variable that can be detected and measured (the observation) which may be related to the object of interest through a calculation. The common analogy given to describe this is trying to determine the type of animal from its footprints. For example, while it is impossible to directly measure temperatures in the upper atmosphere, it is possible to measure the spectral emissions from a known chemical species (such as carbon dioxide) in that region. The frequency of the emissions may then be related via [thermodynamics] to the temperature in that region.

[Wildfire] risk assessment | Remote sensing | Data processing | Data processing levels

#### Data processing levels
To facilitate the discussion of data processing in practice, several processing "levels" were first defined in 1986 by NASA as part of its [Earth Observing System] and steadily adopted since then, both internally at NASA (e. g.,) and elsewhere (e. g.,); these definitions are:
{| class="wikitable"  
! Level
! Description  
A Level 1 data record is the most fundamental (i. e., highest reversible level) data record that has significant scientific utility, and is the foundation upon which all subsequent data sets are produced. Level 2 is the first level that is directly usable for
most scientific applications; its value is much greater than the lower levels. Level 2 data sets tend to be less voluminous than Level 1 data because they have been reduced temporally, spatially, or spectrally. Level 3 data sets are generally smaller than lower level data sets and thus can be dealt with without incurring a great deal of data handling overhead. These data tend to be generally more useful for many applications. The regular spatial and temporal organization of Level 3 datasets makes it feasible to readily combine data from different sources.  
While these processing levels are particularly suitable for typical satellite data processing pipelines, other data level vocabularies have been defined and may be appropriate for more heterogeneous workflows.  
is a form of satellite data which has been pre-processed for immediate [data analysis], such as [data cube]s for [time series] analysis.

[Wildfire] risk assessment | Remote sensing | Applications

### Applications
[Satellite imagery|Satellite images] provide very useful information to produce statistics on topics closely related to the territory, such as agriculture, forestry or land cover in general. The first large project to apply Landsata 1 images for statistics was LACIE (Large Area Crop Inventory Experiment), run by [NASA], [National Oceanic and Atmospheric Administration|NOAA] and the [United States Department of Agriculture|USDA] in 1974–77. Many other application projects on crop area estimation have followed, including the Italian AGRIT project and the MARS project of the [Joint Research Centre] (JRC) of the [European Commission]. Forest area and deforestation estimation have also been a frequent target of remote sensing projects, the same as land cover and land use  
[Ground truth] or reference data to train and validate image classification require a field survey if we are targeting [Annual plant|annual crops] or individual forest species, but may be substituted by [Aerial photographic and satellite image interpretation|photointerpretation] if we look at wider classes that can be reliably identified on [Aerial photography|aerial photos] or satellite images. It is relevant to highlight that probabilistic sampling is not critical for the selection of training pixels for image classification, but it is necessary for accuracy assessment of the classified images and area estimation. Additional care is recommended to ensure that training and validation datasets are not spatially correlated.  
We suppose now that we have classified images or a [Land cover maps|land cover map] produced by visual photo-interpretation, with a legend of mapped classes that suits our purpose, taking again the example of wheat. The straightforward approach is counting the number of pixels classified as wheat and multiplying by the area of each pixel. Many authors have noticed that [estimator] is that it is generally [Bias of an estimator|biased] because [Type I and type II errors|commission and omission errors] in a [confusion matrix] do not compensate each other  
The main strength of classified satellite images or other indicators computed on satellite images is providing cheap information on the whole target area or most of it. This information usually has a good correlation with the target variable (ground truth) that is usually expensive to observe in an unbiased and accurate way. Therefore, it can be observed on a [Sampling (statistics)|probabilistic sample] selected on an [area sampling frame]. Traditional [survey methodology] provides different methods to combine accurate information on a sample with less accurate, but exhaustive, data for a covariable or [Proxy (statistics)|proxy] that is cheaper to collect.  For agricultural statistics, field surveys are usually required, while photo-interpretation may better for land cover classes that can be reliably identified on aerial photographs or high resolution satellite images. Additional uncertainty can appear because of imperfect reference data (ground truth or similar).  
Some options are: [ratio estimator], [regression estimator], [calibration estimators] and [Small area estimation|small area estimators] Messenger pigeons, kites, rockets and unmanned balloons were also used for early images. With the exception of balloons, these first, individual images were not particularly useful for map making or for scientific purposes.  
Systematic [aerial photography] was developed for military surveillance and reconnaissance purposes beginning in [World War I]. After WWI, remote sensing technology was quickly adapted to civilian applications. This is demonstrated by the first line of a 1941 textbook titled "Aerophotography and Aerosurverying," which stated the following:  
The development of remote sensing technology reached a climax during the [Cold War] with the use of modified combat aircraft such as the [P-51], [P-38], [RB-66] and the [F-4C], or specifically designed collection platforms such as the [Lockheed U-2|U2/TR-1], [SR-71], [A-5 Vigilante|A-5] and the [OV-1] series both in overhead and stand-off collection. A more recent development is that of increasingly smaller sensor pods such as those used by law enforcement and the military, in both manned and unmanned platforms. The advantage of this approach is that this requires minimal modification to a given airframe. Later imaging technologies would include infrared, conventional, Doppler and synthetic aperture radar.  
The development of artificial satellites in the latter half of the 20th century allowed remote sensing to progress to a global scale as of the end of the Cold War. Instrumentation aboard various Earth observing and weather satellites such as [Landsat program|Landsat], the [Nimbus program|Nimbus] and more recent missions such as [RADARSAT] and [Upper Atmosphere Research Satellite|UARS] provided global measurements of various data for civil, research, and military purposes. [Space probe]s to other planets have also provided the opportunity to conduct remote sensing studies in extraterrestrial environments, synthetic aperture radar aboard the [Magellan probe|Magellan] spacecraft provided detailed topographic maps of [Venus], while instruments aboard [Solar and Heliospheric Observatory|SOHO] allowed studies to be performed on the [Sun] and the [solar wind], just to name a few examples.  
Recent developments include, beginning in the 1960s and 1970s, the development of [image processing] of [satellite imagery]. The use of the term "remote sensing" began in the early 1960s when [Evelyn Pruitt] realized that advances in science meant that aerial photography was no longer an adequate term to describe the data streams being generated by new technologies. With assistance from her fellow staff member at the Office of Naval Research, Walter Bailey, she coined the term "remote sensing". Several research groups in [Silicon Valley] including [NASA Ames Research Center], [GTE], and [ESL Inc.] developed [Fourier transform] techniques leading to the first notable enhancement of imagery data. In 1999 the first commercial satellite (IKONOS) collecting very high resolution imagery was launched.

[Wildfire] risk assessment | Remote sensing | Training and education

### Training and education
Remote sensing has a growing relevance in the modern information society. It represents a key technology as part of the aerospace industry and bears increasing economic relevance – new sensors e.g. [TerraSAR-X] and [RapidEye] are developed constantly and the demand for skilled labour is increasing steadily. Furthermore, remote sensing exceedingly influences everyday life, ranging from [weather forecasts] to reports on [climate change] or [natural disasters]. As an example, 80% of the German students use the services of [Google Earth]; in 2006 alone the software was downloaded 100 million times. But studies have shown that only a fraction of them know more about the data they are working with. There exists a huge [Knowledge gap hypothesis|knowledge gap] between the application and the understanding of satellite images.
Remote sensing only plays a tangential role in schools, regardless of the political claims to strengthen the support for teaching on the subject. A lot of the computer software explicitly developed for school lessons has not yet been implemented due to its complexity. Thereby, the subject is either not at all integrated into the curriculum or does not pass the step of an interpretation of analogue images. In fact, the subject of remote sensing requires a consolidation of physics and mathematics as well as [Competence (human resources)|competences] in the fields of media and methods apart from the mere visual interpretation of satellite images.  
Many teachers have great interest in the subject "remote sensing", being motivated to integrate this topic into teaching, provided that the curriculum is considered. In many cases, this encouragement fails because of confusing information. In order to integrate remote sensing in a sustainable manner organizations like the [European Geosciences Union|EGU] or [Digital Earth] encourage the development of [E-learning|learning modules] and [Learning management system|learning portals]. Examples include: FIS – Remote Sensing in School Lessons, Geospektiv, Ychange, or Spatial Discovery, to promote media and method qualifications as well as independent learning.

[Wildfire] risk assessment | Remote sensing | Software

### Software
> Main: Remote sensing software  
Remote sensing data are processed and analyzed with computer software, known as a [remote sensing application]. A large number of proprietary and open source applications exist to process remote sensing data.

[Wildfire] risk assessment | Sensor Observation Service | Overview

## Sensor Observation Service  
### Overview  
The Sensor Observation Service (SOS) is a web service to query real-time sensor data and sensor data time series and is part of the Sensor Web. The offered sensor data consists of data directly from the sensors, which are encoded in the Sensor Model Language (SensorML), and the measured values in the Observations and Measurements (O & M) encoding format. The web service as well as both file formats are open standards and specifications of the same name defined by the Open Geospatial Consortium (OGC).
If the SOS supports the transactional profile (SOS-T), new sensors can be registered on the service interface and measuring values be inserted. A SOS implementation can be used both for data from in-situ as well as remote sensing sensors. Furthermore, the sensors can be either mobile or stationary.
Since 2007, the SOS is an official OGC standard. The advantage of the SOS is that sensor data - of any kind - is available in a standardized format using standardized operations. Thus the web-based access to sensor data is simplified. It also allows easy integration into existing Spatial Data Infrastructures or Geographic Information Systems.
In 2016 OGC approved the SensorThings API standard specification, a new RESTful and JSON-based standard provide functions similar to SOS. As both SensorThings API and SOS are based on the OGC/ISO 19156:2011, the two specifications have been demonstrated in an OGC IoT pilot that they can interoperate with each other.  
The Sensor Observation Service (SOS) is a web service to query real-time sensor data and sensor data time series and is part of the [Sensor Web]. The offered sensor data consists of data directly from the sensors, which are encoded in the Sensor Model Language ([SensorML]), and the measured values in the [Observations and Measurements] (O & M) encoding format. The web service as well as both file formats are open standards and specifications of the same name defined by the [Open Geospatial Consortium] (OGC).  
If the SOS supports the transactional profile (SOS-T), new sensors can be registered on the service interface and measuring values be inserted. A SOS implementation can be used both for data from in-situ as well as remote sensing sensors. Furthermore, the sensors can be either mobile or stationary.  
Since 2007, the SOS is an official [Open Geospatial Consortium|OGC] standard. The advantage of the SOS is that sensor data - of any kind - is available in a standardized format using standardized operations. Thus the web-based access to sensor data is simplified. It also allows easy integration into existing [Spatial Data Infrastructure]s or [Geographic Information Systems].  
In 2016 [Open Geospatial Consortium|OGC] approved the [SensorThings API] standard specification, a new RESTful and JSON-based standard provide functions similar to SOS. As both [SensorThings API] and SOS are based on the [Observations and Measurements|OGC/ISO 19156:2011], the two specifications have been demonstrated in an OGC IoT pilot that they can interoperate with each other.

[Wildfire] risk assessment | Sensor Observation Service | Operations

### Operations  
The SOS has three so-called core operations that must be provided by each implementation. The GetCapabilities operation allows you to query a service for a description of the service interface and the available sensor data. For using the SOS, the GetObservation function is probably the most important. It can be utilized to retrieve data for specific sensors. The DescribeSensor function returns detailed information about a sensor or a sensor system and the producing processes.

[Wildfire] risk assessment | Sensor Observation Service | Operations | Core operations (core profile)

#### Core operations (core profile)  
* GetCapabilities returns an [XML] service description with information about the interface (offered operations and endpoints) as well as the available sensor data, such as the period for which sensor data is available, sensors that produce the measured values, or phenomena that are observed (for example air temperature).
* GetObservation allows pull-based querying of observed values, including their metadata. The measured values and their metadata is returned in the [Observations and Measurements] format (O & M).
* DescribeSensor - provides sensor metadata in [SensorML]. The sensor description can contain information about the sensor in general, the identifier and classification, position and observed phenomena, but also details such as calibration data.

[Wildfire] risk assessment | Sensor Observation Service | Operations | Transactional operations (transactional profile)

#### Transactional operations (transactional profile)  
* RegisterSensor allows to register a new sensor in a deployed SOS.
* InsertObservation can be used to insert data for already registered sensors in the SOS.

[Wildfire] risk assessment | Sensor Observation Service | Operations | Extended operations (enhanced profile)

#### Extended operations (enhanced profile)  
* GetResult provides the ability to query for sensor readings without the metadata given consistent metadata (e.g. sensor, observed object).
* GetFeatureOfInterest returns the geoobject whose properties are monitored by sensors in [Geography Markup Language] encoding.
* GetFeatureOfInterestTime provides time periods in which measurements of an observed object in the SOS are available.
* DescribeFeatureType returns the type of the observed geoobjects ([XML Schema (W3C)|XML Schema])
* DescribeObservationType returns the type of observation ([XML Schema (W3C)|XML Schema]), such as om: Measurement).
* GetObservationById allows to query a specific observation using an identifier returned by the service as response to an InsertObservation operation.
* DescribeResultModel provides the [XML Schema (W3C)|XML Schema] of the measured value, which is particularly important for complex measurements, such as multi-spectral data.
