# Version minimale de CMake nécessaire pour comprendre ce projet.
cmake_minimum_required(VERSION 3.20)

# Nom du projet et langage utilisé.
project(
    MonocularVisualOdometryAndMapping
    VERSION 0.1.0
    LANGUAGES CXX
)

# Nous imposons C++20.
# Ainsi le projet utilisera C++20 quel que soit le réglage par défaut
# de Visual Studio sur la machine qui compile le projet.
set(CMAKE_CXX_STANDARD 20)

# Interdit à CMake de retomber silencieusement sur une ancienne norme
# si le compilateur ne supportait pas C++20.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Désactive les extensions propres à certains compilateurs.
# Notre code reste ainsi plus portable.
set(CMAKE_CXX_EXTENSIONS OFF)


# ------------------------------------------------------------
# Recherche des bibliothèques externes.
#
# Grâce à vcpkg, CMake saura où les trouver.
# REQUIRED signifie :
# "si la bibliothèque n'est pas trouvée, arrête la configuration".
# ------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


# ------------------------------------------------------------
# Création de notre programme exécutable.
#
# Le premier argument est le nom de l'exécutable.
# Le second indique quel fichier source doit être compilé.
# ------------------------------------------------------------
add_executable(
    monocular_visual_odometry
    src/main.cpp
)


# ------------------------------------------------------------
# Relie notre programme aux bibliothèques nécessaires.
#
# PRIVATE signifie ici que ces dépendances sont utilisées
# directement par cet exécutable.
# ------------------------------------------------------------
target_link_libraries(
    monocular_visual_odometry
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)


# ------------------------------------------------------------
# Camera calibration utilities
# ------------------------------------------------------------
add_executable(
    camera_calibration
    tools/camera_calibration.cpp
)

target_link_libraries(
    camera_calibration
    PRIVATE
        ${OpenCV_LIBS}
)


# ------------------------------------------------------------
# Video frame extraction utility
# ------------------------------------------------------------
add_executable(
    extract_video_frames
    tools/extract_video_frames.cpp
)

target_link_libraries(
    extract_video_frames
    PRIVATE
        ${OpenCV_LIBS}
)


# ------------------------------------------------------------
# Video monocular visual odometry
# ------------------------------------------------------------
add_executable(
    video_visual_odometry
    src/video_visual_odometry.cpp
)

target_link_libraries(
    video_visual_odometry
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)


# ------------------------------------------------------------
# Trajectory visualization utility
# ------------------------------------------------------------
add_executable(
    visualize_trajectory
    tools/visualize_trajectory.cpp
)

target_link_libraries(
    visualize_trajectory
    PRIVATE
        ${OpenCV_LIBS}
)


# ------------------------------------------------------------
# Map-based monocular visaul odometry
# ------------------------------------------------------------
add_executable(
    map_visual_odometry
    src/map_visual_odometry.cpp
)

target_link_libraries(
    map_visual_odometry
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)