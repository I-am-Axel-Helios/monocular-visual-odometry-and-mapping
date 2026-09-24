#include <opencv2/core/utils/logger.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    // ------------------------------------------------------------
    // Réduit les messages internes d'OpenCV.
    //
    // On garde uniquement les vraies erreurs OpenCV.
    // Les messages INFO concernant FFMPEG, GStreamer, TBB,
    // MSMF, etc. ne seront plus affichés.
    // ------------------------------------------------------------
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_ERROR
    );

    
	std::cout << "\n=== Video Monocular Visual Odometry ===\n\n";

    // ------------------------------------------------------------
    // Arguments
    //
    // argv[1] : vidéo
    // argv[2] : calibration caméra
    // argv[3] : fichier CSV de trajectoire
    // argv[4] : intervalle temporel entre deux images utilisées
    //
    // Exemple :
    //
    // video_visual_odometry
    //     sequence_main.mp4
    //     camera_calibration_video_1920x1080.yaml
    //     trajectory.csv
    //     0.5
    // ------------------------------------------------------------
    if (argc < 4)
    {
        std::cerr << "Usage: video_visual_odometry "
                  << "<video_path> "
                  << "<calibration_file> "
                  << "<output_csv> "
                  << "[sampling interval seconds]\n";

        return 1;
    }

    const fs::path videoPath = argv[1];
    const fs::path calibrationPath = argv[2];
    const fs::path outputCsvPath = argv[3];

    // 0.5 seconde par défaut
    double samplingIntervalSeconds = 0.5;

    if (argc >= 5)
    {
        try
        {
            samplingIntervalSeconds = std::stod(argv[4]);
        }
        catch (...)
        {
            std::cerr << "Error: invalid sampling interval.\n";
            return 1;
        }
    }

    if (samplingIntervalSeconds <= 0.0)
    {
        std::cerr << "Error: sampling interval must be positive.\n";
        return 1;
    }


    // ------------------------------------------------------------
    // Chargement de la calibration vidéo
    // ------------------------------------------------------------
    cv::FileStorage calibrationFile(
        calibrationPath.string(),
        cv::FileStorage::READ
    );


    if (!calibrationFile.isOpened()) 
    {
        std::cerr << "Error: could not open calibration file: "
                  << calibrationPath
                  << '\n';

        return 1;
    }

    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;

    int calibrationWidth = 0;
    int calibrationHeight = 0;


    calibrationFile["camera_matrix"]
        >> cameraMatrix;

    calibrationFile["distortion_coefficients"]
        >> distortionCoefficients;

    calibrationFile["image_width"]
        >> calibrationWidth;

    calibrationFile["image_height"]
        >> calibrationHeight;

    calibrationFile.release();

    if (cameraMatrix.empty() ||
        distortionCoefficients.empty() ||
        cameraMatrix.rows != 3 ||
        cameraMatrix.cols != 3)
    {
        std::cerr << "Error: invalid camera calibration.\n";
        return 1;
    }

    std::cout << "Calibration resolution: "
              << calibrationWidth
              << " x "
              << calibrationHeight
              << "\n\n";


    // ------------------------------------------------------------
    // Ouverture de la vidéo
    // ------------------------------------------------------------
    cv::VideoCapture video(
        videoPath.string()
    );


    if (!video.isOpened())
    {
        std::cerr << "Error: could not open video: "
                  << videoPath
                  << '\n';

        return 1;
    }

    const double fps = video.get(cv::CAP_PROP_FPS);
    const int frameCount = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_COUNT)
    );

    const int videoWidth = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_WIDTH)
    );

    const int videoHeight = static_cast<int>(
        video.get(cv::CAP_PROP_FRAME_HEIGHT)
    );


    if (fps <= 0.0)
    {
        std::cerr << "Error: invalide video FPS.\n";
        return 1;
    }


    std::cout << "Video resolution: "
              << videoWidth
              << " x "
              << videoHeight
              << '\n';

    std:: cout << "FPS: "
               << fps
               << '\n';

    std::cout << "Frame count: "
              << frameCount
              << '\n';

    // La vidéo de calibration et la vidéo de travail 
    // doivent être de même résolution.
    if (videoWidth != calibrationWidth ||
        videoHeight != calibrationHeight)
    {
        std::cerr << "Error: video resolution does not match "
                  << "the calibration resolution.\n";

        return 1;
    }

    // Conversion de l'intervalle temporel en nombre de frames
    // 
    // Exemple:
    //  20 fps * 0.5 s = 10 frames
    const int frameStep = std::max(
        1, static_cast<int>(
                std::round(
                    fps * samplingIntervalSeconds
                )
            )
    );

    std::cout << "Sampling interval: "
              << samplingIntervalSeconds
              << " s\n";

    std::cout << "Frame step: "
              << frameStep
              << "\n\n";


    // ------------------------------------------------------------
    // Préparation du fichier de trajectoire
    // ------------------------------------------------------------
    if (outputCsvPath.has_parent_path())
    {
        fs::create_directories(
            outputCsvPath.parent_path()
        );
    }

    std::ofstream trajectoryFile(
        outputCsvPath
    );

    if (!trajectoryFile.is_open())
    {
        std::cerr << "Error: could not create trajectory file.\n";
        return 1;
    }


    trajectoryFile << "frame_index,"
                   << "time_seconds,"
                   << "x, "
                   << "y, "
                   << "z, "
                   << "good_matches,"
                   << "pose_inliers\n";


    // ------------------------------------------------------------
    // ORB + matcher
    // 
    // On augmente à 2000 pts par rapport au mode photo car les 
    // vidéos sont en 1920x1080. 
    // ------------------------------------------------------------
    cv::Ptr<cv::ORB> orb =
        cv::ORB::create(2000);

    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    const float ratioThreshold = 0.75f;


    // ------------------------------------------------------------
    // Etat de la frame précédente acceptée
    // ------------------------------------------------------------
    cv::Mat previousGray;
    cv::Mat previousDescriptors;

    std::vector<cv::KeyPoint> previousKeypoints;

    bool previousFrameInitialized = false;


    // ------------------------------------------------------------
    // Pose globale de la caméra
    // 
    // Rwc : rotation du repère caméra vers le repère monde.
    // Cw : position du centre de la caméra dans le repère monde.
    // 
    // La première caméra définit notre origine: 
    //      Rwc = I et Cw = 0
    // ------------------------------------------------------------
    cv::Mat worldFromCameraRotation = 
        cv::Mat::eye(
            3,
            3,
            CV_64F
        );

    cv::Mat cameraPositionWorld =
        cv::Mat::zeros(
            3,
            1,
            CV_64F
        );


    // ------------------------------------------------------------
    // Lecture séquentielle de la vidéo
    // ------------------------------------------------------------
    cv::Mat frame;

    int frameIndex = 0;
    int selectedFrames = 0;
    int successfulPoseSteps = 0;

    while (true)
    {
        const bool frameRead =
            video.read(frame);

        if (!frameRead)
        {
            std::cout
                << "\n[VIDEO] Reading stopped at frame "
                << frameIndex
                << " / "
                << frameCount
                << '\n';

            break;
        }

        // On ne traite qu'une frame sur frameStep.
        if (frameIndex % frameStep != 0)
        {
            ++frameIndex;
            continue;
        }

        ++selectedFrames;

        cv::Mat gray;
        cv::cvtColor(
            frame, 
            gray,
            cv::COLOR_BGR2GRAY
        );


        // Détection ORB sur la frame courante
        std::vector<cv::KeyPoint> currentKeypoints;
        cv::Mat currentDescriptors;

        orb->detectAndCompute(
            gray,
            cv::noArray(),
            currentKeypoints,
            currentDescriptors
        );

        if (currentDescriptors.empty())
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : no descriptors\n";

            ++frameIndex;
            continue;
        }

        // Première frame: simple initialisation de la trajectoire
        if (!previousFrameInitialized)
        {
            previousGray = gray.clone();
            previousKeypoints = currentKeypoints;
            previousDescriptors = currentDescriptors.clone();
            previousFrameInitialized = true;

            trajectoryFile << frameIndex
                           << ","
                           << static_cast<double>(frameIndex) / fps
                           << ",0,0,0,0,0\n";

            std::cout << "[INIT] frame "
                      << frameIndex
                      << " : "
                      << currentKeypoints.size()
                      << " keypoints\n";

            ++frameIndex;
            continue;
        }

        // ---------------------------------------------------------------
        // Matching entre la dernière frame acceptée et la frame courante
        // ---------------------------------------------------------------
        std::vector<std::vector<cv::DMatch>> knnMatches;

        matcher.knnMatch(
            previousDescriptors,
            currentDescriptors,
            knnMatches,
            2
        );


        std::vector<cv::DMatch> goodMatches;

        for (const auto& candidates : knnMatches)
        {
            if (candidates.size() < 2)
            {
                continue;
            }

            if (candidates[0].distance < ratioThreshold * candidates[1].distance)
            {
                goodMatches.push_back(
                    candidates[0]
                );
            }
        }


        // On demande un minimum de correspondances pour éviter 
        // une estimation géométrique extrêmement fragile.
        if (goodMatches.size() < 20)
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : only "
                      << goodMatches.size()
                      << " good matches\n";

            ++frameIndex;
            continue;
        }

        // ------------------------------------------------------------
        // Coordonnées des correspondances
        // ------------------------------------------------------------
        std::vector<cv::Point2f> pointsPrevious;
        std::vector<cv::Point2f> pointsCurrent;


        pointsPrevious.reserve(
            goodMatches.size()
        );

        pointsCurrent.reserve(
            goodMatches.size()
        );


        for (const cv::DMatch& match : goodMatches)
        {
            pointsPrevious.push_back(
                previousKeypoints[match.queryIdx].pt
            );

            pointsCurrent.push_back(
                currentKeypoints[match.trainIdx].pt
            );
        }

        // ------------------------------------------------------------
        // Correction de la distorsion
        // ------------------------------------------------------------
        std::vector<cv::Point2f> undistortedPrevious;
        std::vector<cv::Point2f> undistortedCurrent;

        cv::undistortPoints(
            pointsPrevious,
            undistortedPrevious,
            cameraMatrix,
            distortionCoefficients,
            cv::noArray(),
            cameraMatrix
        );


        cv::undistortPoints(
            pointsCurrent,
            undistortedCurrent,
            cameraMatrix,
            distortionCoefficients,
            cv::noArray(),
            cameraMatrix
        );


        // -------------------------------------------------------------
        // Matrice essentielle
        // 
        // Contrairement au démonstrateur à 2 vues, on ne recalculera
        // pas F.
        // 
        // La caméra étant calibrée, findEssentialMat() fait directement
        // le RANSAC géométrique dont on a besoin pour la VO.
        // -------------------------------------------------------------
        cv::Mat essentialMask;

        cv::Mat essentialMatrix =
            cv::findEssentialMat(
                undistortedPrevious,
                undistortedCurrent,
                cameraMatrix,
                cv::RANSAC,
                0.999,
                1.0,
                essentialMask
            );

        if (essentialMatrix.empty())
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : essential matrix failed\n";

            ++frameIndex;
            continue;
        }



        // -------------------------------------------------------------
        // Pose relative
        // -------------------------------------------------------------
        cv::Mat relativeRotation;
        cv::Mat relativeTranslation;
        cv::Mat poseMask = essentialMask.clone();

        const int poseInliers = 
            cv::recoverPose(
                essentialMatrix,
                undistortedPrevious,
                undistortedCurrent,
                cameraMatrix,
                relativeRotation,
                relativeTranslation,
                poseMask
            );

        if (poseInliers < 15)
        {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : only "
                      << poseInliers
                      << " pose inliers\n";

            ++frameIndex;
            continue;
        }


        // -------------------------------------------------------------
        // Composition de la trajectoire
        //
        // recoverPose donne :
        //
        // X_current = R * X_previous + t
        //
        // Le centre de la caméra courante exprimé dans le repère
        // précédent vaut :
        //
        // C = -R^T t
        //
        // On le transforme ensuite dans notre repère monde.
        // -------------------------------------------------------------
        const cv::Mat relativeCameraCenter = 
            -relativeRotation.t() * relativeTranslation;

        cameraPositionWorld +=
            worldFromCameraRotation * relativeCameraCenter;

        worldFromCameraRotation =
            worldFromCameraRotation * relativeRotation.t();

        ++successfulPoseSteps;


        // -------------------------------------------------------------
        // Sauvegarde de la trajectoire
        // -------------------------------------------------------------
        const double x =
            cameraPositionWorld.at<double>(0, 0);

        const double y =
            cameraPositionWorld.at<double>(1, 0);

        const double z =
            cameraPositionWorld.at<double>(2, 0);

        trajectoryFile << frameIndex
                       << ","
                       << static_cast<double>(frameIndex) / fps
                       << ","
                       << x
                       << ","
                       << y
                       << "," 
                       << z
                       << ","
                       << goodMatches.size()
                       << ","
                       << poseInliers
                       << '\n';


        std::cout << "[OK] frame "
                  << frameIndex
                  << " | matches: "
                  << goodMatches.size()
                  << " | pose inliers: "
                  << poseInliers
                  << " | position: ["
                  << x
                  << ","
                  << y
                  << ","
                  << z
                  << "]\n";


        // -------------------------------------------------------------
        // Actualisation de la frame de référence (la frame courante 
        // devient la frame de référence)
        // -------------------------------------------------------------
        previousGray = gray.clone();
        previousKeypoints = currentKeypoints;
        previousDescriptors = currentDescriptors.clone();

        ++frameIndex;
    }

    video.release();
    trajectoryFile.close();



    // -------------------------------------------------------------
    // Résumé
    // -------------------------------------------------------------
    std::cout << "\n=== VO summary ===\n";

    std::cout << "Selected frames: "
              << selectedFrames
              << '\n';

    std::cout << "Successful pose steps: "
                << successfulPoseSteps
                << '\n';

    std::cout << "Trajectory saved to: "
                << outputCsvPath.string()
                << '\n';

    std::cout << "Last frame index reached: "
              << frameIndex
              << '\n';


    return 0;
}

