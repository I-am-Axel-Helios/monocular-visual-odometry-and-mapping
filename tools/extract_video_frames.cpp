#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // ---------------------------------------------------------------
    // Arguments du programme
    // 
    // argv[0] = nom du programme
    // argv[1] = chemin de la vidéo
    // argv[2] = dossier de sortie
    // argv[3] = intervalle entre deux images
    // 
    // Exemple: extract_video_frames calibration_video.mp4 frames 1.0
    // ---------------------------------------------------------------
    if (argc < 4) {
        std::cerr << "Usage: camera_calibration "
                  << "<video_path> "
                  << "<output_directory> "
                  << "<interval seconds>\n";

        return 1;
    }

    const fs::path videoPath = argv[1];
    const fs::path outputDirectory = argv[2];

    double intervalSeconds = 0.0;

    try {
        intervalSeconds = std::stod(argv[3]);
    } catch (...) {
        std::cerr << "Error: interval_seconds must be a number.\n";
        return 1;
    }

    if (intervalSeconds <= 0.0) {
        std::cerr << "Error: interval_seconds must be positive.\n";
        return 1;
    }


    // ---------------------------------------------------------------
    // Ouverture de la vidéo
    // ---------------------------------------------------------------
    cv::VideoCapture video(
        videoPath.string()
    );

    if (!video.isOpened()) {
        std::cerr << "Error: could not open video: "
                  << videoPath
                  << "\n";

        return 1;
    }


    // ---------------------------------------------------------------
    // Lecture des métadonnées principales
    // ---------------------------------------------------------------
    const double fps = video.get(cv::CAP_PROP_FPS);

    const int frameCount = 
        static_cast<int>(
            video.get(cv::CAP_PROP_FRAME_COUNT)
        );

    const int width =
        static_cast<int>(
            video.get(cv::CAP_PROP_FRAME_WIDTH)
        );

    const int height =
        static_cast<int>(
            video.get(cv::CAP_PROP_FRAME_HEIGHT)
        );


    if (fps <= 0.0) {
        std::cerr << "Error: invalid video FPS.\n";
        return 1;
    }


    const double durationSeconds =
        static_cast<double>(frameCount) / fps;


    std::cout << "\n=== Video information ===\n";

    std::cout << "Resolution: "
              << width
              << " x "
              << height
              << '\n';

    std::cout << "FPS: "
              << fps
              << '\n';

    std::cout << "Frame count: "
              << frameCount
              << '\n';

    std::cout << "Approximate duration: "
              << durationSeconds
              << " s\n";

    std::cout << "Extraction interval: "
              << intervalSeconds
              << " s\n\n";


    // ---------------------------------------------------------------
    // Création du dossier de sortie
    // ---------------------------------------------------------------
    fs::create_directories(
        outputDirectory
    );


    // ---------------------------------------------------------------
    // Nombre approximatif de frames à sauter
    //
    // Exemple :
    //
    // vidéo 30 fps
    // intervalle = 1 seconde
    //
    // frameStep ≈ 30
    // ---------------------------------------------------------------
    const int frameStep =
        std::max(
            1,
            static_cast<int>(
                std::round(fps * intervalSeconds)
                )
        );

    std::cout << "Frame step: "
              << frameStep
              << " frames\n\n";

    // ---------------------------------------------------------------
    // Lecture séquentielle de la vidéo
    // ---------------------------------------------------------------
    cv::Mat frame;

    int currentFrameIndex = 0;
    int savedFrameIndex = 0;

    while (video.read(frame)) {
        // On ne sauvegarde qu'une frame tous les frameStep
        if (currentFrameIndex % frameStep == 0) {
            std::ostringstream filename;
            // frame_0000.jpg
            // frame_0001.jpg
            //  ...
            filename << "frame_"
                     << std::setw(4)
                     << std::setfill('0')
                     << savedFrameIndex
                     << ".jpg";


            const fs::path outputPath =
                outputDirectory / filename.str();

            const bool success = 
                cv::imwrite(
                    outputPath.string(),
                    frame
                );

            if (!success) {
                std::cerr << "Error: could not save "
                          << outputPath
                          << '\n';

                return 1;
            }


            std::cout << "[SAVED] "
                      << filename.str()
                      << " <- video frame "
                      << currentFrameIndex
                      << '\n';

            ++savedFrameIndex;
        }

        ++currentFrameIndex;
    }

    video.release();

    // ------------------------------------------------------------
    // Résultat final
    // ------------------------------------------------------------
    std::cout << "\n=== Extraction complete ===\n";

    std::cout << "Frames read: "
              << currentFrameIndex
              << '\n';

    std::cout << "Frames saved: "
              << savedFrameIndex
              << '\n';

    std::cout << "Output directory: "
              << outputDirectory.string()
              << '\n';

    return 0;
}