#include <opencv2/core/utils/logger.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <array>

#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace fs = std::filesystem;

// ----------------------------------------------------------------
// Informations de qualité associées à un point 3D de la carte.
// 
// Cette structure reste parallèle à mapPoints:
//      mapPoints[i], mapDescriptors.row(i) et mapPointQualities[i]
//      représentent toujours le même point physique.
// ----------------------------------------------------------------
struct MapPointQuality {
    // Frame à laquelle le point 3D a été triangulé
    int creationFrameIndex = -1;

    // Dernière frame dans laquelle ce point a été validé
    // comme inlier PnP
    // À sa création, on considère qu'il vient d'être vu.
    int lastSeenFrameIndex = -1;

    // Nombre de fois où le descripteur de ce point a produit une correspondance
    // 3D-2D candidate pour PnP
    int pnpMatchCount = 0;

    // Nombre de fois où cette correspondance a réellement survécu au RANSAC PnP
    int pnpInlierCount = 0;

    // Dernière frame dans laquelle le descripteur du point
    // a produit une correspondance 3D-2D après Lowe + déduplication 2D.
    // 
    // ATTENTION: cela ne signifie PAS que le point était un inlier PnP.
    int lastMatchedFrameIndex = -1;
};

using MapPointId = std::uint64_t;

// valeur réservée signifiant "aucun point 3D valide"
constexpr MapPointId invalidMapPointId =    
    std::numeric_limits<MapPointId>::max();

// -----------------------------------------------------------------
// Observation d'un point 3D dans une keyframe.
// 
// mapPointId est stable même si le culling compacte mapPoints.
// -----------------------------------------------------------------
struct KeyframeObservation {
    MapPointId mapPointId = invalidMapPointId;

    cv::Point2d imagePoint;
};

// ----------------------------------------------------------------
// Infos nécessaires au futur Bundle Adjustment
// 
// Convention: X_camera = Rcw * X_world + tcw
// ----------------------------------------------------------------
struct KeyframeRecord {
    int frameIndex = -1;

    cv::Mat worldToCameraRotation;
    cv::Mat translationVector;

    std::vector<KeyframeObservation> observations;
};


// --------------------------------------------------------------------------
// Pose locale utilisée par le Bundle Adjustment
// 
// parameters[0..2] : rotation angle-axis
// parameters[3..5] : translation tcw
// 
// Convention: X_camera = Rcw * X_world + tcw
// --------------------------------------------------------------------------
struct LocalBundleAdjustmentPose {
    int frameIndex = -1;

    std::array<double, 6> parameters{};


    // Les deux plus anciennes poses seront fixées pour supprimer
    // les libertés de jauge du problème monoculaire.
    bool fixed = false;
};


// --------------------------------------------------------------------------
// Point 3D actif inclus dans la fenêtre locale
// --------------------------------------------------------------------------
struct LocalBundleAdjustmentPoint {
    MapPointId mapPointId = invalidMapPointId;

    // Indice actuel dans mapPoints
    // 
    // Valable que pour l'état courant de la carte
    int globalMapPointIndex = -1;

    std::array<double, 3> position{};
};


// ----------------------------------------------------------------------
// Mesure 2D reliant une pose locale à un point 3D local
// ----------------------------------------------------------------------
struct LocalBundleAdjustmentObservation {
    int poseIndex= -1;
    int pointIndex = -1;

    cv::Point2d imagePoint;
};


// ------------------------------------------------------------------------
// Problème complet envoyé plus tard à l'optimiseur
// ------------------------------------------------------------------------
struct LocalBundleAdjustmentProblem {
    std::vector<LocalBundleAdjustmentPose> poses;
    std::vector<LocalBundleAdjustmentPoint> points;
    std::vector<LocalBundleAdjustmentObservation> observations;
};

// ------------------------------------------------------------------
// Résidu de reprojection utilisé par Ceres pour le Bundle
// Adjustment local.
// 
// Paramètres caméra:
//      camera[0..2] = rotation angle-axis
//      camera[3..5] = translation tcw
// 
// Paramètres point:
//      point[0..2] = coordonnées 3D dans le repère monde
// 
// Convention: X_camera = Rcw * X_world + tcw
// 
// Les observations 2D fournies à ce résidu sont déjà UNDISTORTED
// On utilise donc un modèle pinhole simple, sans k1, k2, p1, p2 ...
// ------------------------------------------------------------------
struct BundleAdjustmentReprojectionResidual {
    BundleAdjustmentReprojectionResidual(
        double observedX, double observedY,
        double fx, double fy,
        double cx, double cy) :
        observedX(observedX), observedY(observedY),
        fx(fx), fy(fy), cx(cx), cy(cy) {
    }

    template <typename T> bool operator()(
        const T* const camera,
        const T* const point,
        T* residuals) const
    {
        // -----------------------------------------------------------
        // Rotation du point monde vers le repère caméra.
        // 
        // camera[0..2] est une rotation angle-axis.
        // -----------------------------------------------------------
        T pointCamera[3];

        ceres::AngleAxisRotatePoint(
            camera, point, pointCamera
        );


        // ------------------------------------------------------------
        // Translation monde -> caméra
        // ------------------------------------------------------------
        pointCamera[0] += camera[3];
        pointCamera[1] += camera[4];
        pointCamera[2] += camera[5];


        // -----------------------------------------------------------
        // Projection perspective pinhole
        // 
        // xn = Xc/Zc & yn = Yc/Zc
        // -----------------------------------------------------------
        const T normalizedX =
            pointCamera[0] / pointCamera[2];

        const T normalizedY =
            pointCamera[1] / pointCamera[2];

        // -----------------------------------------------------------
        // Passage aux coordonnées pixels
        // -----------------------------------------------------------
        const T projectedX =
            T(fx) * normalizedX + T(cx);

        const T projectedY =
            T(fy) * normalizedY + T(cy);


        // -----------------------------------------------------------
        // Résidu = projection prédite - observation mésurée
        // 
        // Ceres cherchera à rendre ces deux valeurs aussi proches de 
        // zéro que possible.
        // -----------------------------------------------------------
        residuals[0] =
            projectedX - T(observedX);

        residuals[1] =
            projectedY - T(observedY);

        return true;
    }

    double observedX = 0.0;
    double observedY = 0.0;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};


int cullOldNeverConfirmedMapPoints(
    int currentFrameIndex,
    int minimumPointAgeFrames,
    int maximumUnmatchedAgeFrames,

    std::vector<cv::Point3f>& mapPoints,
    cv::Mat& mapDescriptors,
    std::vector<MapPointQuality>& mapPointQualities,
    std::vector<MapPointId>& mapPointIds,

    std::vector<int>& keyframeMapPointIndices,
    std::vector<int>& currentMapPointIndices) {
    // ----------------------------------------------------
    // Sécurité: les 3 structures de carte doivent être 
    // prafaitement synchronisées avant toute suppression.
    // ----------------------------------------------------
    if (mapPoints.size() !=
            static_cast<std::size_t>(mapDescriptors.rows) ||
        mapPoints.size() !=
            mapPointQualities.size() ||
        mapPoints.size() !=
            mapPointIds.size()
        ) {
        std::cerr << "Error: inconsistant map before culling.\n";
        return -1;
    }


    const std::size_t oldMapSize = mapPoints.size();

    // oldToNew[i] = 
    //      nouvel indice du point i s'il est conservé,
    //      -1 s'il est supprimé
    std::vector<int> oldToNew(
        oldMapSize, -1
    );

    std::vector<cv::Point3f> newMapPoints;
    std::vector<MapPointQuality> newMapPointQualities;
    cv::Mat newMapDescriptors;
    std::vector<MapPointId> newMapPointIds;

    newMapPoints.reserve(oldMapSize);
    newMapPointQualities.reserve(oldMapSize);
    newMapPointIds.reserve(oldMapSize);

    int removedPoints = 0;
    int protectedByRecentMatch = 0;

    for (std::size_t oldIndex = 0; oldIndex < oldMapSize; ++oldIndex) {
        const MapPointQuality& quality = mapPointQualities[oldIndex];

        const int pointAgeFrames = currentFrameIndex - quality.creationFrameIndex;

        const bool oldEnough =
            pointAgeFrames >= minimumPointAgeFrames;

        const bool neverConfirmed =
            quality.pnpInlierCount == 0;

        // Le point est considéré récemment retrouvé s'il a produit 
        // un bon match dans la fenêtre temporelle récente.
        const bool recentlyMatched =
            quality.lastMatchedFrameIndex >= 0 &&
            (currentFrameIndex - quality.lastMatchedFrameIndex) < maximumUnmatchedAgeFrames;

        const bool shouldCull =
            oldEnough &&
            neverConfirmed &&
            !recentlyMatched;

        if (oldEnough && neverConfirmed && recentlyMatched) {
            ++protectedByRecentMatch;
        }

        if (shouldCull) {
            ++removedPoints;
            continue;
        }

        const int newIndex =
            static_cast<int>(
                newMapPoints.size()
            );

        oldToNew[oldIndex] = newIndex;

        newMapPoints.push_back(
            mapPoints[oldIndex]
        );

        newMapPointQualities.push_back(quality);

        newMapDescriptors.push_back(
            mapDescriptors.row(
                static_cast<int>(oldIndex)
            )
        );

        newMapPointIds.push_back(
            mapPointIds[oldIndex]
        );
    }

    // -------------------------------------------------------------
    // Les associations keypoint -> mapPoint utilisent les anciens
    // indices. Il faut donc impérativement les remapper.
    // -------------------------------------------------------------
    auto remapAssociations =
        [&oldToNew](std::vector<int>& associations) {
            for (int& mapPointIndex : associations) {
                if (mapPointIndex < 0) {
                    continue;
                }

                if (mapPointIndex >=
                    static_cast<int>(oldToNew.size())) {
                    mapPointIndex = -1;
                    continue;
                }

                mapPointIndex = oldToNew[mapPointIndex];
            }
    };

    remapAssociations(
        keyframeMapPointIndices
    );

    remapAssociations(
        currentMapPointIndices
    );


    // Installation de la carte compactée
    mapPoints.swap(
        newMapPoints
    );

    mapPointQualities.swap(
        newMapPointQualities
    );

    mapPointIds.swap(
        newMapPointIds
    );

    mapDescriptors = newMapDescriptors;

    std::cout << "[MAP CULLING]"
              << " removed: "
              << removedPoints
              << " | protected by recent match: "
              << protectedByRecentMatch
              << " | size: "
              << oldMapSize
              << " -> "
              << mapPoints.size()
              << '\n';

    return removedPoints;
}


int addNewMapPointsBetweenKeyframes(
    const std::vector<cv::KeyPoint>& keyframeKeypoints,
    const cv::Mat& keyframeDescriptors,
    std::vector<int>& keyframeMapPointIndices,
    const cv::Mat& keyframeWorldToCameraRotation,
    const cv::Mat& keyframeTranslationVector,

    const std::vector<cv::KeyPoint>& currentKeypoints,
    const cv::Mat& currentDescriptors,
    std::vector<int>& currentMapPointIndices,
    const cv::Mat& currentWorldToCameraRotation,
    const cv::Mat& currentTranslationVector,

    const cv::Mat& cameraMatrix,
    const cv::Mat& distortionCoefficients,

    float ratioThreshold,

    std::vector<cv::Point3f>& mapPoints,
    cv::Mat& mapDescriptors) {
    // -----------------------------------------------------------
    // Matching entre ancienne keyframe et nouvelle keyframe
    // 
    // On cherche ici des features qui ne correspondent PAS encore
    // à des points 3D existants
    // -----------------------------------------------------------
    cv::BFMatcher matcher(
        cv::NORM_HAMMING,
        false
    );

    std::vector<std::vector<cv::DMatch>> knnMatches;

    matcher.knnMatch(
        keyframeDescriptors,
        currentDescriptors,
        knnMatches,
        2
    );

    std::vector<cv::DMatch> candidateMatches;

    for (const auto& candidates : knnMatches) {
        if (candidates.size() < 2) {
            continue;
        }

        const cv::DMatch& best = candidates[0];
        const cv::DMatch& second = candidates[1];

        // Lowe ratio test
        if (best.distance >= ratioThreshold * second.distance) {
            continue;
        }

        // Le point de l'ancienne keyframe appartient déjà à la carte:
        // inutile de le trianguler à nouveau
        if (keyframeMapPointIndices[best.queryIdx] >= 0) {
            continue;
        }

        // Le point courant appartient déjà à la carte.
        if (currentMapPointIndices[best.trainIdx] >= 0) {
            continue;
        }

        candidateMatches.push_back(
            best
        );
    }
    // On privilégie les meilleurs matches avantt de suppimer les
    // duplications éventuelles sur l'image courante.
    std::sort(
        candidateMatches.begin(),
        candidateMatches.end(),
        [](const cv::DMatch& a,
            const cv::DMatch& b) {
            return a.distance < b.distance;
        }
    );

    std::vector<bool> currentKeypointAlreadyUsed(
        currentKeypoints.size(),
        false
    );

    std::vector<cv::DMatch> triangulationMatches;

    for (const cv::DMatch& match : candidateMatches) {
        if (currentKeypointAlreadyUsed[match.trainIdx]) {
            continue;
        }

        currentKeypointAlreadyUsed[match.trainIdx] = true;

        triangulationMatches.push_back(match);
    }

    if (triangulationMatches.size() < 20) {
        return 0;    
    }

    // --------------------------------------------------------------
    // Coordonnées 2D correspondant aux nouveaux matches
    // --------------------------------------------------------------
    std::vector<cv::Point2f> pointsKeyframe;
    std::vector<cv::Point2f> pointsCurrent;

    for (const cv::DMatch& match : triangulationMatches) {
        pointsKeyframe.push_back(
            keyframeKeypoints[match.queryIdx].pt
        );
        pointsCurrent.push_back(
            currentKeypoints[match.trainIdx].pt
        );
    }

    // --------------------------------------------------------------
    // Correction de la distorsion
    // 
    // P = cameraMatrix signifie que les coordonnées résultantes
    // restent exprimées en pixels.
    // --------------------------------------------------------------
    std::vector<cv::Point2f> undistortedKeyframe;
    std::vector<cv::Point2f> undistortedCurrent;

    cv::undistortPoints(
        pointsKeyframe,
        undistortedKeyframe,
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

    // ------------------------------------------------------------
    // Matrices de projection ABSOLUES
    // 
    // Contrairement au bootstrap initial:
    // 
    // P1 = K [ R1 | t1 ]
    // P2 = K [ R2 | t2 ]
    // 
    // Les points triangulés seront donc directement exprimés dans
    // le repère monde de notre carte.
    // ------------------------------------------------------------
    cv::Mat keyframeExtrinsic;
    cv::Mat currentExtrinsic;

    cv::hconcat(
        currentWorldToCameraRotation,
        currentTranslationVector,
        currentExtrinsic
    );

    cv::hconcat(
        keyframeWorldToCameraRotation,
        keyframeTranslationVector,
        keyframeExtrinsic
    );

    const cv::Mat projectionKeyframe = 
        cameraMatrix * keyframeExtrinsic;

    const cv::Mat projectionCurrent =
        cameraMatrix * currentExtrinsic;


    // ------------------------------------------------------------
    // Centres des deux caméras dans le repère monde
    // 
    // Pour une pose: X_camera = Rcw * X_world + tcw
    // 
    // Centre caméra dans le monde: Cw = -Rcw^T * tcw
    // ------------------------------------------------------------
    const cv::Mat keyframeCameraCenterWorld = 
        -keyframeWorldToCameraRotation.t() * keyframeTranslationVector;

    const cv::Mat currentCameraCenterWorld = 
        -currentWorldToCameraRotation.t() * currentTranslationVector;


    // ------------------------------------------------------------
    // Angle minimal de triangulation
    // 
    // Si les 2 rayons de vue sont presque parallèles, la 
    // profondeur du point devient très mal contrainte.
    // ------------------------------------------------------------
    const double minimumTriangulationAngleDegrees = 0.25;


    // ------------------------------------------------------------
    // Triangulation
    // ------------------------------------------------------------
    cv::Mat homogeneousPoints4D;

    cv::triangulatePoints(
        projectionKeyframe,
        projectionCurrent,
        undistortedKeyframe,
        undistortedCurrent,
        homogeneousPoints4D
    );

    cv::Mat homogeneousPoints4D64;
    homogeneousPoints4D.convertTo(
        homogeneousPoints4D64,
        CV_64F
    );


    std::vector<double> triangulationAngles;

    // -----------------------------------------------------------
    // Validation et ajout à la carte
    // 
    // On impose :
    //      - coordonnées finies;
    //      - profondeur poisitive dans les deux caméras;
    //      - erreur de reprojection <= 2 px dans les deux vues.
    // -----------------------------------------------------------
    const double maximumReprojectionError = 2.0;

    int addedPoints = 0;
    int rejectedByLowParallax = 0;

    for (int i = 0; i < homogeneousPoints4D64.cols; ++i) {
        const double w =
            homogeneousPoints4D64.at<double>(
                3,i
            );

        if (std::abs(w) < 1e-9) {
            continue;
        }

        const double pointX = 
            homogeneousPoints4D64.at<double>(
                0, i
            ) / w;

        const double pointY =
            homogeneousPoints4D64.at<double>(
                1, i
            ) / w;

        const double pointZ =
            homogeneousPoints4D64.at<double>(
                2, i
            ) / w;

        if (!std::isfinite(pointX) ||
            !std::isfinite(pointY) ||
            !std::isfinite(pointZ)) {
            continue;
        }

        cv::Mat pointWorld = 
            (cv::Mat_<double>(3, 1)
                << pointX,
                   pointY,
                   pointZ);

        cv::Mat pointKeyframe = 
            keyframeWorldToCameraRotation * pointWorld + keyframeTranslationVector;

        cv::Mat pointCurrent =
            currentWorldToCameraRotation * pointWorld + currentTranslationVector;

        // Point derrière une caméra
        if (pointKeyframe.at<double>(2, 0) <= 0.0 ||
            pointCurrent.at<double>(2, 0) <= 0.0) {
            continue;
        }
        
        // --------------------------------------------------------
        // Contrôle de parallaxe
        //
        // On construit les deux rayons allant des centres caméra
        // vers le point 3D.
        //
        //                 X
        //                / \
        //               /   \
        //             C1     C2
        //
        // L'angle entre ces deux rayons doit être suffisamment
        // grand pour que la profondeur soit bien contrainte.
        // --------------------------------------------------------
        cv::Mat rayFromKeyframe = 
            pointWorld - keyframeCameraCenterWorld;

        cv::Mat rayFromCurrent =
            pointWorld - currentCameraCenterWorld;

        const double keyframeRayNorm = 
            cv::norm(rayFromKeyframe);

        const double currentRayNorm =
            cv::norm(rayFromCurrent);

        // Protection contre les cas dégénérés
        if (keyframeRayNorm < 1e-9 ||
            currentRayNorm < 1e-9) {
            continue;
        }

        // Le cosinus d'un angle theta est donné par:
        // cos(theta) = (a.b) / (||a||  ||b||)
        double cosineAngle = 
            rayFromKeyframe.dot(rayFromCurrent) / (keyframeRayNorm * currentRayNorm);

        // Les petites erreurs numériques peuvent produire par exemple
        // 1.0000000001, ce qui rendrait acos() invalide.
        cosineAngle = 
            std::clamp(
                cosineAngle,
                -1.0,
                1.0
            );

        const double triangulationAngleDegrees =
            std::acos(cosineAngle) * 180.0 / CV_PI;


        triangulationAngles.push_back(
            triangulationAngleDegrees
        );

        if (triangulationAngleDegrees < minimumTriangulationAngleDegrees) {
            ++rejectedByLowParallax;
            continue;
        }


        // -------------------------------------------------------
        // Reprojection dans la keyframe
        // -------------------------------------------------------
        cv::Mat projectedKeyframe = cameraMatrix * pointKeyframe;

        const cv::Point2d reprojectionKeyframe(
            projectedKeyframe.at<double>(0, 0) / projectedKeyframe.at<double>(2, 0),
            projectedKeyframe.at<double>(1, 0) / projectedKeyframe.at<double>(2, 0)
        );

        // -------------------------------------------------------
        // Reprojection dans la frame courante
        // -------------------------------------------------------
        cv::Mat projectedCurrent = cameraMatrix * pointCurrent;

        const cv::Point2d reprojectionCurrent(
            projectedCurrent.at<double>(0, 0) / projectedCurrent.at<double>(2, 0),
            projectedCurrent.at<double>(1, 0) / projectedCurrent.at<double>(2, 0)
        );


        const double errorKeyframe = 
            cv::norm(
                reprojectionKeyframe - cv::Point2d(undistortedKeyframe[i])
            );

        const double errorCurrent =
            cv::norm(
                reprojectionCurrent - cv::Point2d(undistortedCurrent[i])
            );


        if (errorKeyframe > maximumReprojectionError ||
            errorCurrent > maximumReprojectionError) {
            continue;
        }


        // -------------------------------------------------------
        // Nouveau point valide
        // -------------------------------------------------------
        const cv::DMatch& match = triangulationMatches[i];
        const int newMapPointIndex = 
            static_cast<int>(
                mapPoints.size()
            );

        mapPoints.emplace_back(
            static_cast<float>(pointX),
            static_cast<float>(pointY),
            static_cast<float>(pointZ)
        );

        // On conserve le descripteur de l'observation la plus récente.
        mapDescriptors.push_back(
            currentDescriptors.row(match.trainIdx)
        );

        // Association dans les deux keyframes
        keyframeMapPointIndices[match.queryIdx] = newMapPointIndex;
        currentMapPointIndices[match.trainIdx] = newMapPointIndex;

        ++addedPoints;
    }

    std::cout << "[TRIANGULATION] "
              << "matches: "
              << triangulationMatches.size()
              << " | low-parallax rejected: "
              << rejectedByLowParallax
              << " | added: "
              << addedPoints
              << '\n';


    if (!triangulationAngles.empty()) {
        std::sort(
            triangulationAngles.begin(),
            triangulationAngles.end()
        );

        const double minimumAngle = 
            triangulationAngles.front();

        const double medianAngle = 
            triangulationAngles[triangulationAngles.size() / 2];

        const double maximumAngle =
            triangulationAngles.back();

        int above025 = 0;
        int above050 = 0;
        int above100 = 0;

        for (double angle : triangulationAngles) {
            if (angle >= 0.25) {
                ++above025;
            }

            if (angle >= 0.50) {
                ++above050;
            }

            if (angle >= 1.00) {
                ++above100;
            }
        }

        std::cout << "[PARALLAX] min: "
                  << minimumAngle
                  << " deg"
                  << " | median: "
                  << medianAngle
                  << " | max: "
                  << maximumAngle
                  << " deg\n";

        std::cout << "[PARALLAX] >=0.25: "
                  << above025
                  << " | >=0.50: "
                  << above050
                  << " | >= 1.00: "
                  << above100
                  << '\n';
    }

    return addedPoints;
}


bool createKeyframeRecord(
    int frameIndex,

    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<int>& keypointMapPointIndices,

    const std::vector<MapPointId>& mapPointIds,

    const cv::Mat& worldToCameraRotation,
    const cv::Mat& translationVector,

    KeyframeRecord& record) {
    // Une association est définie pour chacun des keypoints.
    if (keypoints.size() != keypointMapPointIndices.size()) {
        std::cerr << "Error: keyframe keypoint/association mismatch.\n";
        return false;
    }

    record = KeyframeRecord{};

    record.frameIndex = frameIndex;

    record.worldToCameraRotation = 
        worldToCameraRotation.clone();

    record.translationVector = 
        translationVector.clone();

    record.observations.reserve(
        keypoints.size()
    );


    for (std::size_t keypointIndex = 0;
        keypointIndex < keypoints.size();
        ++keypointIndex) {
        const int mapPointIndex = 
            keypointMapPointIndices[keypointIndex];

        // Ce keypoint n'a aucune association 3D.
        if (mapPointIndex < 0) {
            continue;
        }

        if (mapPointIndex >=
            static_cast<int>(mapPointIds.size())
            ) {
            std::cerr << "Error: invalid map point index "
                      << "while rerecording keyframe"
                      << " | frame: "
                      << frameIndex
                      << " | map index: "
                      << mapPointIndex
                      << " | map size: "
                      << mapPointIds.size()
                      << '\n';

            return false;
        }


        KeyframeObservation observation;

        observation.mapPointId = 
            mapPointIds[mapPointIndex];

        observation.imagePoint = 
            cv::Point2d(
                static_cast<double>(
                    keypoints[keypointIndex].pt.x
                ),
                static_cast<double>(
                    keypoints[keypointIndex].pt.y
                )
            );

        record.observations.push_back(
            observation
        );
    }

    return true;
}

bool refreshStoredKeyframeRecord(
    std::vector<KeyframeRecord>& keyframeHistory,

    int frameIndex,

    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<int>& keypointMapPointIndices,

    const std::vector<MapPointId>& mapPointIds,

    const cv::Mat& worldToCameraRotation,
    const cv::Mat& translationVector) {
    auto keyframeIterator =
        std::find_if(
            keyframeHistory.rbegin(),
            keyframeHistory.rend(),

            [frameIndex](const KeyframeRecord& record) {
                return record.frameIndex == frameIndex;
            }
        );

    if (keyframeIterator == keyframeHistory.rend()) {
        std::cerr << "Error: mapping anchor frame "
                  << frameIndex
                  << " not found in keyframe history.\n";

        return false;
    }


    KeyframeRecord refreshedRecord;

    if (!createKeyframeRecord(
        frameIndex,

        keypoints,
        keypointMapPointIndices,

        mapPointIds,

        worldToCameraRotation,
        translationVector,

        refreshedRecord)) {
        return false;
    }

    std::cout << "[KEYFRAME REFRESH] frame: "
              << frameIndex
              << " | refreshed observations: "
              << refreshedRecord.observations.size()
              << '\n';

    *keyframeIterator = std::move(refreshedRecord);

    return true;
}

void printLocalBundleAdjustmentConnectivity(
    const std::vector<KeyframeRecord>& keyframeHistory,
    const std::vector<MapPointId>& activeMapPointIds,
    std::size_t windowSize) {
    if (keyframeHistory.empty()) {
        return;
    }

    const std::size_t actualWindowSize = 
        std::min(
            windowSize,
            keyframeHistory.size()
        );

    const std::size_t firstKeyframeIndex = 
        keyframeHistory.size() - actualWindowSize;

    std::unordered_set<MapPointId> activeMapPointIdSet(
            activeMapPointIds.begin(),
            activeMapPointIds.end()
        );


    // Nombre de keyframes de la fenêtre dans lesquelles chaque point 3D est observé
    std::unordered_map<MapPointId, int> observationCounts;

    for (std::size_t keyframeIndex = firstKeyframeIndex;
        keyframeIndex < keyframeHistory.size();
        ++keyframeIndex) {
        const KeyframeRecord& keyframe =
            keyframeHistory[keyframeIndex];

        for (const KeyframeObservation& observation : keyframe.observations) {
            // Une ancienne keyframe peut encore contenir l'ID d'un point qui
            // a depuis été supprimé de la carte.
            // 
            // Ces observations historiques ne peuvent pas participer au Bundle 
            // Adjustment de la carte active.
            if (!activeMapPointIdSet.contains(observation.mapPointId)) {
                continue;
            }
            
            ++observationCounts[
                observation.mapPointId
            ];
        }
    }

    int observedAtLeastOnce = 0;
    int observedAtLeastTwice = 0;
    int observedAtLeastThreeTimes = 0;
    int observedAtLeastFourTimes = 0;
    int observedInEveryKeyframe = 0;

    for (const auto& [mapPointId, count] : observationCounts) {
        (void)mapPointId;

        if (count >= 1) ++observedAtLeastOnce;
        if (count >= 2) ++observedAtLeastTwice;
        if (count >= 3) ++observedAtLeastThreeTimes;
        if (count >= 4) ++observedAtLeastFourTimes;
        if (count >=
            static_cast<int>(actualWindowSize)) {
            ++observedInEveryKeyframe;
        }
    }

    std::cout << "[LOCAL BA WINDOW] keyframes: "
              << actualWindowSize
              << " | frames: ";

    for (std::size_t keyframeIndex = firstKeyframeIndex;
        keyframeIndex < keyframeHistory.size();
        ++keyframeIndex) {
        std::cout << keyframeHistory[keyframeIndex].frameIndex;

        if (keyframeIndex + 1 < keyframeHistory.size()) {
            std::cout << ",";
        }
    }

    std::cout << " | unique points: "
              << observedAtLeastOnce
              << " | seen >= 2: "
              << observedAtLeastTwice
              << " | seen >= 3: "
              << observedAtLeastThreeTimes
              << " | seen >= 4: "
              << observedAtLeastFourTimes
              << " | seen in all: "
              << observedInEveryKeyframe
              << '\n';
}


bool buildLocalBundleAdjustmentProblem(
    const std::vector<KeyframeRecord>& keyframeHistory,

    const std::vector<cv::Point3f>& mapPoints,
    const std::vector<MapPointId>& mapPointIds,

    const std::unordered_set<MapPointId>& excludedFromLocalBundleAdjustment,

    const cv::Mat& cameraMatrix,
    const cv::Mat& distortionCoefficients,

    std::size_t windowSize,

    LocalBundleAdjustmentProblem& problem) {
    problem = LocalBundleAdjustmentProblem{};

    // ---------------------------------------------------------------
    // Vérif de cohérence de la carte active
    // ---------------------------------------------------------------
    if (mapPoints.size() != mapPointIds.size()) {
        std::cerr << "Error: point/ID mismatch while building local BA\n";
        return false;
    }

    if (keyframeHistory.size() < windowSize) {
        return true;
    }

    const std::size_t firstKeyframeIndex = 
        keyframeHistory.size() - windowSize;

    // ----------------------------------------------------------------
    // ID stable -> indice courant dans mapPoints
    // 
    // Ceci permet de retrouver la position 3D actuelle d'un MapPointId
    // historique
    // ----------------------------------------------------------------
    std::unordered_map<MapPointId, int> activeMapPointIndices;

    activeMapPointIndices.reserve(
        mapPointIds.size()
    );

    for (std::size_t mapPointIndex = 0;
        mapPointIndex < mapPointIds.size();
        ++mapPointIndex) {
        const MapPointId mapPointId = mapPointIds[mapPointIndex];

        const auto [iterator, inserted] = 
            activeMapPointIndices.emplace(
                mapPointId,
                static_cast<int>(mapPointIndex)
            );

        (void)iterator;

        if (!inserted) {
            std::cerr << "Error: duplicate MapPointId "
                      << mapPointId
                      << " in active map.\n";

            return false;
        }
    }


    // ---------------------------------------------------------------------
    // Première passe:
    // 
    // combien de keyframes différentes observent chaque point ?
    // 
    // Un point vu dans une seule keyframe n'est pas retenu pour le BA local
    // ---------------------------------------------------------------------
    std::unordered_map<MapPointId, int> observationCounts;

    for (std::size_t historyIndex = firstKeyframeIndex;
        historyIndex < keyframeHistory.size();
        ++historyIndex) {
        const KeyframeRecord& keyframe = 
            keyframeHistory[historyIndex];


        // Protection supplémentaire contre une éventuelle observation
        // dupliquée dans une même keyframe
        std::unordered_set<MapPointId> seenInThisKeyframe;

        for (const KeyframeObservation& observation : keyframe.observations) {
            if (!activeMapPointIndices.contains(observation.mapPointId)) {
                continue;
            }

            if (excludedFromLocalBundleAdjustment.contains(observation.mapPointId)) {
                continue;
            }

            if (!seenInThisKeyframe.insert(observation.mapPointId).second) {
                continue;
            }

            ++observationCounts[observation.mapPointId];
        }
    }


    // ------------------------------------------------------------------
    // Construction des 5 poses locales
    // ------------------------------------------------------------------
    problem.poses.reserve(windowSize);

    for (std::size_t localPoseIndex = 0; localPoseIndex < windowSize; ++localPoseIndex) {
        const KeyframeRecord& keyframe = 
            keyframeHistory[firstKeyframeIndex + localPoseIndex];

        cv::Mat rotation64;
        cv::Mat translation64;

        keyframe.worldToCameraRotation.convertTo(rotation64, CV_64F);
        keyframe.translationVector.convertTo(translation64, CV_64F);


        // Conversion matrice de rotation -> vecteur angle-axis.
        // 
        // OpenCV appelle cette représentation un "rotation vector"
        cv::Mat rotationVector;

        cv::Rodrigues(rotation64, rotationVector);

        LocalBundleAdjustmentPose pose;

        pose.frameIndex = keyframe.frameIndex;

        pose.parameters[0] = rotationVector.at<double>(0, 0);
        pose.parameters[1] = rotationVector.at<double>(1, 0);
        pose.parameters[2] = rotationVector.at<double>(2, 0);
        pose.parameters[3] = translation64.at<double>(0, 0);
        pose.parameters[4] = translation64.at<double>(1, 0);
        pose.parameters[5] = translation64.at<double>(2, 0);


        // Deux poses fixes:
        // Cela va préserver le repère & l'échelle déjà établis par notre V0 monoculaire
        pose.fixed = localPoseIndex < 2;

        problem.poses.push_back(pose);
    }

    // --------------------------------------------------------------------------
    // Construction des points ayant au moins 2 observations.
    // 
    // On parcourt mapPointIds plutôt que l'unordered_map pour conserver un ordre
    // deterministe entre les exécutions.
    // --------------------------------------------------------------------------
    std::unordered_map<MapPointId, int> localPointIndices;

    for (std::size_t globalPointIndex = 0; globalPointIndex < mapPointIds.size(); ++globalPointIndex) {
        const MapPointId mapPointId = mapPointIds[globalPointIndex];

        if (excludedFromLocalBundleAdjustment.contains(mapPointId)) {
            continue;
        }

        const auto countIterator = observationCounts.find(mapPointId);

        if (countIterator == observationCounts.end()) {
            continue;
        }

        if (countIterator->second < 2) {
            continue;
        }

        LocalBundleAdjustmentPoint point;

        point.mapPointId = mapPointId;
        point.globalMapPointIndex = static_cast<int>(globalPointIndex);

        point.position[0] = static_cast<double>(mapPoints[globalPointIndex].x);
        point.position[1] = static_cast<double>(mapPoints[globalPointIndex].y);
        point.position[2] = static_cast<double>(mapPoints[globalPointIndex].z);

        const int localPointIndex = static_cast<int>(problem.points.size());

        localPointIndices.emplace(mapPointId, localPointIndex);
        problem.points.push_back(point);
    }


    // ---------------------------------------------------------------------
    // Deuxième passe: construction des arêtes 2D du graphe BA
    // ---------------------------------------------------------------------
    for (std::size_t localPoseIndex = 0;
        localPoseIndex < windowSize;
        ++localPoseIndex) {
        const KeyframeRecord& keyframe = 
            keyframeHistory[
                firstKeyframeIndex + localPoseIndex
            ];

        std::unordered_set<MapPointId> seenInThisKeyframe;

        for (const KeyframeObservation& observation : keyframe.observations) {
            const auto pointIterator = 
                localPointIndices.find(observation.mapPointId);

            if (pointIterator == localPointIndices.end()) {
                continue;
            }

            if (!seenInThisKeyframe.insert(observation.mapPointId).second) {
                continue;
            }

            LocalBundleAdjustmentObservation localObservation;

            localObservation.poseIndex = 
                static_cast<int>(localPoseIndex);

            localObservation.pointIndex =
                pointIterator->second;

            std::vector<cv::Point2f> distortedPoints{
                cv::Point2f(
                    static_cast<float>(observation.imagePoint.x),
                    static_cast<float>(observation.imagePoint.y)
                )
            };

            std::vector<cv::Point2f> undistortedPoints;

            cv::undistortPoints(
                distortedPoints,
                undistortedPoints,
                cameraMatrix,
                distortionCoefficients,
                cv::noArray(),
                cameraMatrix
            );

            if (undistortedPoints.size() != 1) {
                std::cerr << "Error: failed to undistort BA observation.\n";
                return false;
            }


            localObservation.imagePoint = 
                cv::Point2d(
                    static_cast<double>(undistortedPoints[0].x),
                    static_cast<double>(undistortedPoints[0].y)
                );


            problem.observations.push_back(localObservation);
        }
    }


    // -------------------------------------------------------------------
    // Diagnostic par pose
    // -------------------------------------------------------------------
    std::vector<int> observationsPerPose(
        problem.poses.size(), 0
    );

    for (const LocalBundleAdjustmentObservation& observation : problem.observations) {
        ++observationsPerPose[observation.poseIndex];
    }

    std::cout << "[LOCAL BA PROBLEM] poses: "
              << problem.poses.size()
              << " | points: "
              << problem.points.size()
              << " | observations: "
              << problem.observations.size()
              << " | fixed poses: 2"
              << " | observations/pose: ";

    for (std::size_t poseIndex = 0;
        poseIndex < observationsPerPose.size();
        ++poseIndex) {
        std::cout << problem.poses[poseIndex].frameIndex
                  << ":"
                  << observationsPerPose[poseIndex];

        if (poseIndex + 1 < observationsPerPose.size()) {
            std::cout << ",";
        }
    }

    std::cout << '\n';

    return true;
}

bool runLocalBundleAdjustment(LocalBundleAdjustmentProblem& localProblem, const cv::Mat& cameraMatrix) {

    if (localProblem.poses.empty() || localProblem.points.empty() || localProblem.observations.empty()) {
        std::cerr << "Error: empty local BA problem.\n";
        return false;
    }

    if (cameraMatrix.rows != 3 || cameraMatrix.cols != 3) {
        std::cerr << "Error: invalid camera matrix for local BA.\n";
        return false;
    }

    // ------------------------------------------------------------
    // Intrinsèques de la caméra
    // ------------------------------------------------------------
    cv::Mat cameraMatrix64;
    cameraMatrix.convertTo(cameraMatrix64, CV_64F);

    const double fx = cameraMatrix64.at<double>(0, 0);
    const double fy = cameraMatrix64.at<double>(1, 1);
    const double cx = cameraMatrix64.at<double>(0, 2);
    const double cy = cameraMatrix64.at<double>(1, 2);


    // -------------------------------------------------------------
    // Création du problème Ceres
    // -------------------------------------------------------------
    ceres::Problem ceresProblem;


    // -------------------------------------------------------------
    // Ajout explicite des poses
    // 
    // Chaque pose = 6 params:
    //      angle-axis (3) & translation (3)
    // -------------------------------------------------------------
    for (LocalBundleAdjustmentPose& pose : localProblem.poses) {
        ceresProblem.AddParameterBlock(pose.parameters.data(), 6);

        // Fixation des deux plus anciennes poses
        if (pose.fixed) {
            ceresProblem.SetParameterBlockConstant(pose.parameters.data());
        }
    }

    // --------------------------------------------------------------
    // Points 3D
    // --------------------------------------------------------------
    for (LocalBundleAdjustmentPoint& point : localProblem.points) {
        ceresProblem.AddParameterBlock(point.position.data(), 3);
    }

    // --------------------------------------------------------------
    // Huber loss
    // 
    // Une observation ayant une grosse erreur de reprojection aura 
    // moins d'influence sur l'optimisation.
    // 
    // Delta = 2 px
    // --------------------------------------------------------------
    ceres::LossFunction* huberLoss = new ceres::HuberLoss(2.0);


    // --------------------------------------------------------------
    // Une observation 2D = un bloc résiduel de dimension 2
    // --------------------------------------------------------------
    std::vector<ceres::ResidualBlockId> residualBlockIds;
    residualBlockIds.reserve(localProblem.observations.size());

    for (const LocalBundleAdjustmentObservation& observation : localProblem.observations) {
        if (observation.poseIndex < 0 ||
            observation.poseIndex >= static_cast<int>(localProblem.poses.size()) ||
            observation.pointIndex < 0 ||
            observation.pointIndex >= static_cast<int>(localProblem.points.size())) {

            std::cerr << "Error: invalid local BA observation index.\n";
            return false;
        }

        LocalBundleAdjustmentPose& pose = localProblem.poses[observation.poseIndex];
        LocalBundleAdjustmentPoint& point = localProblem.points[observation.pointIndex];

        ceres::CostFunction* costFunction =
            new ceres::AutoDiffCostFunction<BundleAdjustmentReprojectionResidual, 2, 6, 3>(
                new BundleAdjustmentReprojectionResidual(
                    observation.imagePoint.x, observation.imagePoint.y,
                    fx, fy,
                    cx, cy
                )
            );

        const ceres::ResidualBlockId residualBlockId = 
            ceresProblem.AddResidualBlock(
                costFunction, huberLoss, pose.parameters.data(), point.position.data()
            );

        residualBlockIds.push_back(residualBlockId);
    }

    // ---------------------------------------------------------------
    // Mesure AVANT optimisation
    // 
    // Ici, on désactive volontairement la Huber Loss afin de calculer
    // une vraie erreur quadratique de reprojection.
    // 
    // Ceres définit: cost = 1/2 * somme(résidu²)
    // Donc RMSE_2D = sqrt(2 cost/nbre_observations)
    // ---------------------------------------------------------------
    ceres::Problem::EvaluateOptions evaluateOptions;

    evaluateOptions.apply_loss_function = false;

    double initialRawCost = 0.0;

    if (!ceresProblem.Evaluate(
            evaluateOptions, &initialRawCost,
            nullptr, nullptr, nullptr)
       ) {
        std::cerr << "Error: could not evaluate local BA before solving.\n";
        return false;
    }


    const double initialRmse =
        std::sqrt(
            2.0 * initialRawCost / static_cast<double>(localProblem.observations.size())
        );

    if (residualBlockIds.size() != localProblem.observations.size()) {
        std::cerr << "Error: residual block / observation mismatch"
                  << " | residual blocks: "
                  << residualBlockIds.size()
                  << " | observations: "
                  << localProblem.observations.size()
                  << '\n';

        return false;
    }

    auto printPerPoseReprojectionDiagnostics = [&](const char* label) {
        
        std::vector<double> squaredErrorSum(localProblem.poses.size(), 0.0);
        std::vector<double> maximumError(localProblem.poses.size(), 0.0);

        std::vector<int> observationCount(localProblem.poses.size(), 0);
        std::vector<int> above2Pixels(localProblem.poses.size(), 0);
        std::vector<int> above5Pixels(localProblem.poses.size(), 0);
        std::vector<int> above10Pixels(localProblem.poses.size(), 0);

        for (std::size_t observationIndex = 0;
             observationIndex < localProblem.observations.size();
             ++observationIndex) {

            double residualCost = 0.0;
            double residuals[2] = {0.0, 0.0};

            const bool evaluationSucceeded = ceresProblem.EvaluateResidualBlock(
                residualBlockIds[observationIndex], false, &residualCost, residuals, nullptr
            );

            if (!evaluationSucceeded) {
                continue;
            }

            const double error = 
                std::sqrt(residuals[0]*residuals[0] + residuals[1]*residuals[1]);

            if (!std::isfinite(error)) {
                continue;
            }

            const int poseIndex = localProblem.observations[observationIndex].poseIndex;
            
            if (poseIndex < 0 || poseIndex >= static_cast<int>(localProblem.poses.size())) {
                continue;
            }


            squaredErrorSum[poseIndex] += error * error;
            maximumError[poseIndex] = 
                std::max(maximumError[poseIndex], error);
            ++observationCount[poseIndex];

            if (error >= 2.0) {
                ++above2Pixels[poseIndex];
            }

            if (error >= 5.0) {
                ++above5Pixels[poseIndex];
            }

            if (error >= 10.0) {
                ++above10Pixels[poseIndex];
            }
        }

        for (std::size_t poseIndex = 0; poseIndex < localProblem.poses.size(); ++poseIndex) {

            const int count = observationCount[poseIndex];
            const double rmse = 
                count > 0 ? std::sqrt(squaredErrorSum[poseIndex] / static_cast<double>(count)) : 0.0;

            std::cout << "[LOCAL BA PER-POSE " 
                      << label
                      << "] frame: "
                      << localProblem.poses[poseIndex].frameIndex
                      << " | fixed: "
                      << (localProblem.poses[poseIndex].fixed ? "yes" : "no")
                      << " | observations: "
                      << count
                      << " | RMSE: "
                      << rmse
                      << " px"
                      << " | max: "
                      << maximumError[poseIndex]
                      << " px"
                      << " | >= 2px: "
                      << above2Pixels[poseIndex]
                      << " | >= 5px: "
                      << above5Pixels[poseIndex]
                      << " | >= 10px: "
                      << above10Pixels[poseIndex]
                      << '\n';
        }
    };

    printPerPoseReprojectionDiagnostics("BEFORE");


    // ---------------------------------------------------------------
    // Sauvegarde de l'état initial
    // 
    // Pour l'instant, le BA travailletoujours sur une copie locale.
    // On conserve ici les paraètres initiaux afin de mesurer
    // précisément les corrections produites par Ceres.
    // ---------------------------------------------------------------
    std::vector<std::array<double, 6>> initialPoseParameters;
    initialPoseParameters.reserve(localProblem.poses.size());

    for (const LocalBundleAdjustmentPose& pose : localProblem.poses) {
        initialPoseParameters.push_back(pose.parameters);
    };


    std::vector<std::array<double, 3>> initialPointPositions;
    initialPointPositions.reserve(localProblem.points.size());

    for (const LocalBundleAdjustmentPoint& point : localProblem.points) {
        initialPointPositions.push_back(point.position);
    };

    // ---------------------------------------------------------------
    // Configuration du solveur
    // 
    // Il n'y a que 5 poses dans notre fenêtre.
    // DENSE_SCHUR est donc très adapté et simple pour ce premier BA.
    // ---------------------------------------------------------------
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 30;
    options.num_threads = 1;

    // ---------------------------------------------------------------
    // Optimisation
    // ---------------------------------------------------------------
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ceresProblem, &summary);

    // ---------------------------------------------------------------
    // Mesure APRES optimisation
    // ---------------------------------------------------------------
    double finalRawCost = 0.0;

    if (!ceresProblem.Evaluate(
            evaluateOptions, &finalRawCost,
            nullptr, nullptr, nullptr)
       ) {
        std::cerr << "Error: could not evaluate local BA after solving.\n";
        return false;
    }

    const double finalRmse =
        std::sqrt(
            2.0 * finalRawCost / static_cast<double>(localProblem.observations.size())
        );

    printPerPoseReprojectionDiagnostics("AFTER");

    // ---------------------------------------------------------------
    // Diagnostic des corrections produites par Ceres
    // 
    // Ce bloc mesure les corrections produites par Ceres avant leur
    // éventuelle réinjection dans l'état global.
    // ---------------------------------------------------------------
    bool optimizedStateIsFinite = true;

    // 1. Déplacement des poses
    for (std::size_t poseIndex = 0; poseIndex < localProblem.poses.size(); ++poseIndex) {

         const std::array<double, 6>& before = initialPoseParameters[poseIndex];
         const std::array<double, 6>& after = localProblem.poses[poseIndex].parameters;

         for (double value : after) {
             if (!std::isfinite(value)) {
                 optimizedStateIsFinite = false;
             }
         }

         // Rotation avant optimisation
         cv::Mat rotationVectorBefore =
            (cv::Mat_<double>(3, 1)
                << before[0], before[1], before[2]);
         
         cv::Mat rotationBefore;
         cv::Rodrigues(rotationVectorBefore, rotationBefore);

         // Translation avant optimisation
         cv::Mat translationBefore =
            (cv::Mat_<double>(3, 1)
                << before[3], before[4], before[5]);

         // Centre caméra monde avant: Cw = -Rcw^T * tcw
         const cv::Mat cameraCenterBefore = -rotationBefore.t() * translationBefore;


         // Rotation après optimisation
         cv::Mat rotationVectorAfter =
             (cv::Mat_<double>(3, 1)
                 << after[0], after[1], after[2]);

         cv::Mat rotationAfter;
         cv::Rodrigues(rotationVectorAfter, rotationAfter);

         // Translation après optimisation
         cv::Mat translationAfter =
             (cv::Mat_<double>(3, 1)
                 << after[3], after[4], after[5]);

         // Centre caméra monde après
         const cv::Mat cameraCenterAfter = -rotationAfter.t() * translationAfter;


        // -----------------------------------------------------------
        // Correction de position caméra
        // -----------------------------------------------------------
        const double centerDisplacement =
            cv::norm(cameraCenterAfter - cameraCenterBefore);

        // -----------------------------------------------------------
        // Correction angulaire
        // 
        // Rotation relative entre avant et après
        // -----------------------------------------------------------
        const cv::Mat relativeRotation =
            rotationAfter * rotationBefore.t();

        cv::Mat relativeRotationVector;
        cv::Rodrigues(relativeRotation, relativeRotationVector);

        const double rotationCorrectionDegrees = 
            cv::norm(relativeRotationVector) * 180.0 / CV_PI;

        std::cout << "[LOCAL BA POSE DELTA] frame: "
                  << localProblem.poses[poseIndex].frameIndex
                  << " | fixed: "
                  << (localProblem.poses[poseIndex].fixed ? "yes" : "no")
                  << " | rotation: "
                  << rotationCorrectionDegrees
                  << " deg"
                  << " | center displacement: "
                  << centerDisplacement
                  << " arbitrary units"
                  << '\n';
    }

    // 2. Déplacement des points 3D
    std::vector<double> pointDisplacements;

    pointDisplacements.reserve(
        localProblem.points.size()
    );

    for (std::size_t pointIndex = 0; 
         pointIndex < localProblem.points.size(); 
         ++pointIndex) {
         const std::array<double, 3>& before =
             initialPointPositions[pointIndex];

         const std::array<double, 3>& after =
             localProblem.points[pointIndex].position;

         if (!std::isfinite(after[0]) ||
             !std::isfinite(after[1]) ||
             !std::isfinite(after[2])) {
             optimizedStateIsFinite = false;
             continue;
         }

         const double dx = after[0] - before[0];
         const double dy = after[1] - before[1];
         const double dz = after[2] - before[2];

         const double displacement = 
            std::sqrt(dx*dx + dy*dy + dz*dz);

         pointDisplacements.push_back(
             displacement
        );  
    }

    std::sort(
        pointDisplacements.begin(), pointDisplacements.end()
    );

    double medianPointDisplacement = 0.0;
    double p90PointDisplacement = 0.0;
    double maximumPointDisplacement = 0.0;

    if (!pointDisplacements.empty()) {
        medianPointDisplacement = pointDisplacements[pointDisplacements.size() / 2];

        const std::size_t p90Index =
            static_cast<std::size_t>(
                0.90 * static_cast<double>(pointDisplacements.size() - 1)
            );

        p90PointDisplacement = pointDisplacements[p90Index];
        maximumPointDisplacement = pointDisplacements.back();
    }

    const double maximumAcceptedPointDisplacement =
        std::max(
            1.0, 3.0 * p90PointDisplacement
        );

    std::size_t acceptedPointUpdates = 0;
    std::size_t rejectedPointUpdates = 0;

    for (std::size_t pointIndex = 0; pointIndex < localProblem.points.size(); ++pointIndex) {
        const std::array<double, 3>& before =
            initialPointPositions[pointIndex];

        const std::array<double, 3>& after =
            localProblem.points[pointIndex].position;

        const double dx = after[0] - before[0];
        const double dy = after[1] - before[1];
        const double dz = after[2] - before[2];

        const double displacement = 
            std::sqrt(dx*dx + dy*dy + dz*dz);

        if (std::isfinite(displacement) && displacement <= maximumAcceptedPointDisplacement) {
            ++acceptedPointUpdates;
        }
        else {
            ++rejectedPointUpdates;
        }
    }

    std::cout << "[LOCAL BA UPDATE CANDIDATES] accepted points: "
              << acceptedPointUpdates
              << "/"
              << localProblem.points.size()
              << " | rejected points: "
              << rejectedPointUpdates
              << " | displacement threshold: "
              << maximumAcceptedPointDisplacement
              << '\n';


    std::cout << "[LOCAL BA POINT DELTA] median: "
              << medianPointDisplacement
              << " | p90: "
              << p90PointDisplacement
              << " | max: "
              << maximumPointDisplacement
              << " arbitrary units"
              << '\n';

    // 3. Vérif de la profondeur après optimisation
    // 
    // Un BA peut mathématiquement réduire un coût tout en produisant 
    // une géométrie physiquement absurde.
    // On vérifie donc que les points restent devant les caméras qui 
    // les observent.
    std::size_t invalidDepthObservations = 0;

    for (const LocalBundleAdjustmentObservation& observation : 
            localProblem.observations) {

        const LocalBundleAdjustmentPose& pose =
            localProblem.poses[observation.poseIndex];

        const LocalBundleAdjustmentPoint& point =
            localProblem.points[observation.pointIndex];

        double pointCamera[3];

        ceres::AngleAxisRotatePoint(
            pose.parameters.data(),
            point.position.data(),
            pointCamera
        );


        pointCamera[0] += pose.parameters[3];
        pointCamera[1] += pose.parameters[4];
        pointCamera[2] += pose.parameters[5];

        if (!std::isfinite(pointCamera[0]) ||
            !std::isfinite(pointCamera[1]) ||
            !std::isfinite(pointCamera[2]) ||
            pointCamera[2] <= 1e-6) {
            ++invalidDepthObservations;
        }
    }

    std::cout << "[LOCAL BA SAFETY] finite: "
              << (optimizedStateIsFinite ? "yes" : "no")
              << " | invalid depth observations: "
              << invalidDepthObservations
              << "/"
              << localProblem.observations.size()
              << '\n';

    // ---------------------------------------------------------------
    // Diagnostic
    // ---------------------------------------------------------------
    std::cout << "[LOCAL BA] poses: "
        << localProblem.poses.size()
        << " | points: "
        << localProblem.points.size()
        << " | observations: "
        << localProblem.observations.size()
        << " | RMSE: "
        << initialRmse
        << " -> "
        << finalRmse
        << " px"
        << " | iterations: "
        << summary.iterations.size()
        << " | usable: "
        << (summary.IsSolutionUsable() ? "yes" : "no")
        << '\n';

    return summary.IsSolutionUsable();
}


bool applyLocalBundleAdjustmentResults(
    const LocalBundleAdjustmentProblem& localProblem,

    std::vector<cv::Point3f>& mapPoints,
    const std::vector<MapPointId>& mapPointIds,

    std::unordered_set<MapPointId>& excludedFromLocalBundleAdjustment,

    std::vector<KeyframeRecord>& keyframeHistory,

    int currentFrameIndex,
    cv::Mat& worldToCameraRotation,
    cv::Mat& translationVector,
    cv::Mat& worldFromCameraRotation,
    cv::Mat& cameraPositionWorld,

    int mappingAnchorFrameIndex,
    cv::Mat& keyframeWorldToCameraRotation,
    cv::Mat& keyframeTranslationVector,
    
    double maximumAllowedPoseCenterCorrection) {

    // -------------------------------------------------------------------
    // 1. Vérif des poses optimisées
    // -------------------------------------------------------------------
    for (const LocalBundleAdjustmentPose& pose : localProblem.poses) {
        for (double value : pose.parameters) {
            if (!std::isfinite(value)) {
                std::cerr << "Error: non-finite pose produced by local BA.\n";
                return false;
            }
        }
    }

    // -------------------------------------------------------------------
    // Validation géométrique des corrections de pose AVANT write-back
    // 
    // Une solution Ceres peut être numériquement valide tout en déplaçant 
    // excessivement une pose par rapport à l'état VO courant.
    // Dans ce cas, on rejette tout le write-back de cette fenêtre.
    // -------------------------------------------------------------------
    const double maximumAcceptedRotationCorrectionDegrees = 2.0;

    double maximumRotationCorrectionDegrees = 0.0;
    double currentPoseCenterCorrection = 0.0;

    bool currentPoseFound = false;

    std::vector<cv::Mat> cameraCentersBefore;
    cameraCentersBefore.reserve(localProblem.poses.size());

    for (const LocalBundleAdjustmentPose& localPose : localProblem.poses) {
        const auto historyIterator =
            std::find_if(
                keyframeHistory.begin(), keyframeHistory.end(),
                [&localPose](const KeyframeRecord& record) {
                    return record.frameIndex == localPose.frameIndex;
                }
            );

        if (historyIterator == keyframeHistory.end()) {
            std::cerr << "Error: BA pose not found in keyframe history"
                      << " | frame: "
                      << localPose.frameIndex
                      << '\n';

            return false;
        }

        cv::Mat rotationBefore;
        cv::Mat translationBefore;

        historyIterator->worldToCameraRotation.convertTo(
            rotationBefore, CV_64F
        );

        historyIterator->translationVector.convertTo(
            translationBefore, CV_64F
        );

        cv::Mat rotationVectorAfter =
            (cv::Mat_<double>(3, 1)
                << localPose.parameters[0], localPose.parameters[1], localPose.parameters[2]);

        cv::Mat rotationAfter;
        cv::Rodrigues(
            rotationVectorAfter, rotationAfter
        );

        cv::Mat translationAfter =
            (cv::Mat_<double>(3, 1)
                << localPose.parameters[3], localPose.parameters[4], localPose.parameters[5]);
        
        const cv::Mat cameraCenterBefore = -rotationBefore.t() * translationBefore;
        const cv::Mat cameraCenterAfter = -rotationAfter.t() * translationAfter;

        cameraCentersBefore.push_back(cameraCenterBefore.clone());

        const double centerCorrection = 
            cv::norm(cameraCenterAfter - cameraCenterBefore);

        const cv::Mat relativeRotation = 
            rotationAfter * rotationBefore.t();

        cv::Mat relativeRotationVector;
        cv::Rodrigues(
            relativeRotation, relativeRotationVector
        );

        const double rotationCorrectionDegrees =
            cv::norm(relativeRotationVector) * 180.0 / CV_PI;

        if (!localPose.fixed) {
            maximumRotationCorrectionDegrees = 
                std::max(
                    maximumRotationCorrectionDegrees, rotationCorrectionDegrees
                );
        }

        if (localPose.frameIndex == currentFrameIndex) {
            currentPoseCenterCorrection = centerCorrection;
            currentPoseFound = true;
        }
    }


    if (!currentPoseFound) {
        std::cerr << "Error: current keyframe not found in localBA window.\n";
        return false;
    }

    const bool excessiveRotation =
        maximumRotationCorrectionDegrees > maximumAcceptedRotationCorrectionDegrees;

    const bool excessiveCurrentPoseCorrection =
        std::isfinite(maximumAllowedPoseCenterCorrection) &&
        currentPoseCenterCorrection > maximumAllowedPoseCenterCorrection;

    // -------------------------------------------------------------------
    // 2. Calcul des déplacements des points 3D
    // 
    // mapPoints contient encore ici les positions AVANT BA.
    // localProblem.Points contient celles APRÈS BA.
    // -------------------------------------------------------------------
    std::vector<double> pointDisplacements;
    std::vector<double> relativePointDisplacements;
    std::vector<double> relativePointDisplacementByPoint(
        localProblem.points.size(), std::numeric_limits<double>::quiet_NaN()
    );

    pointDisplacements.reserve(
        localProblem.points.size()
    );

    relativePointDisplacements.reserve(
        localProblem.points.size()
    );

    for (std::size_t localPointIndex = 0;
         localPointIndex < localProblem.points.size();
         ++localPointIndex) {

        const LocalBundleAdjustmentPoint& localPoint = localProblem.points[localPointIndex];
        const int globalIndex = 
            localPoint.globalMapPointIndex;

        if (globalIndex < 0 ||
            globalIndex >= static_cast<int>(mapPoints.size()) ||
            globalIndex >= static_cast<int>(mapPointIds.size())) {

            std::cerr << "Error: invalid global map point indexduring BA write-back.\n";
            return false;
        }

        if (mapPointIds[globalIndex] != localPoint.mapPointId) {
            std::cerr << "Error: map point ID mismatch during BA write-back.\n";
            return false;
        }

        const cv::Point3f& before = mapPoints[globalIndex];
        const std::array<double, 3>& after = localPoint.position;

        if (!std::isfinite(after[0]) ||
            !std::isfinite(after[1]) ||
            !std::isfinite(after[2])) {

            continue;
        }

        const double dx = after[0] - static_cast<double>(before.x);
        const double dy = after[1] - static_cast<double>(before.y);
        const double dz = after[2] - static_cast<double>(before.z);

        const double displacement = std::sqrt(dx*dx + dy*dy + dz*dz);

        pointDisplacements.push_back(displacement);

        // Diagnostic du déplacement relatif.
        //
        // Pour rendre la mesure indépendante de l'échelle monoculaire,
        // on rapporte le déplacement du point à sa distance à la caméra
        // observante la plus proche avant le BA.
        // 
        // ratio = ||X_après - X_avant|| / ||X_avant - C||
        // 
        // Exemple: 
        //      0.10 -> déplacement de 10% de la distance caméra-point
        //      0.5 -> 50 %
        //      1.00 -> 100 %
        const cv::Mat pointBefore = 
            (cv::Mat_<double>(3, 1)
                << static_cast<double>(before.x),
                   static_cast<double>(before.y),
                   static_cast<double>(before.z));

        double minimumObservationDistance = std::numeric_limits<double>::infinity();

        for (const LocalBundleAdjustmentObservation& observation : localProblem.observations) {
            if (observation.pointIndex != static_cast<int>(localPointIndex)) {
                continue;
            }

            if (observation.poseIndex < 0 || observation.poseIndex >= static_cast<int>(cameraCentersBefore.size())) {
                continue;
            }

            const double observationDistance = 
                cv::norm(pointBefore - cameraCentersBefore[observation.poseIndex]);

            if (std::isfinite(observationDistance) && observationDistance > 1e-9) {
                minimumObservationDistance = 
                    std::min(minimumObservationDistance, observationDistance);
            }
        }

        if (std::isfinite(minimumObservationDistance) && minimumObservationDistance > 1e-9) {
            const double relativeDisplacement = displacement / minimumObservationDistance;

            if (std::isfinite(relativeDisplacement)) {
                relativePointDisplacementByPoint[localPointIndex] = relativeDisplacement;
                relativePointDisplacements.push_back(relativeDisplacement);
            }
        }
    }

    if (relativePointDisplacements.size() != localProblem.points.size()) {
        std::cerr << "Error: missing relative displacement "
                  << "for some BA points"
                  << " | relative samples : "
                  << relativePointDisplacements.size()
                  << " | BA points: "
                  << localProblem.points.size()
                  << '\n';
    }

    std::sort(relativePointDisplacements.begin(), relativePointDisplacements.end());

    double medianRelativePointDisplacement = 0.0;
    double p90RelativePointDisplacement = 0.0;
    double maximumRelativePointDisplacement = 0.0;

    if (!relativePointDisplacements.empty()) {
        medianRelativePointDisplacement = relativePointDisplacements[relativePointDisplacements.size() / 2];

        const std::size_t relativeP90Index =
            static_cast<std::size_t>(
                0.90 * static_cast<double>(relativePointDisplacements.size() - 1)
                );

        p90RelativePointDisplacement = relativePointDisplacements[relativeP90Index];
        maximumRelativePointDisplacement = relativePointDisplacements.back();
    }

    std::size_t relativeAbove10Percent = 0;
    std::size_t relativeAbove25Percent = 0;
    std::size_t relativeAbove50Percent = 0;
    std::size_t relativeAbove100Percent = 0;

    for (double relativeDisplacement : relativePointDisplacements) {
        if (relativeDisplacement >= 0.10) {
            ++relativeAbove10Percent;
        }

        if (relativeDisplacement >= 0.25) {
            ++relativeAbove25Percent;
        }

        if (relativeDisplacement >= 0.50) {
            ++relativeAbove50Percent;
        }

        if (relativeDisplacement >= 1.00) {
            ++relativeAbove100Percent;
        }
    }

    std::cout << "[LOCAL BA RELATIVE POINT DELTA] median: "
              << medianRelativePointDisplacement
              << " | p90: "
              << p90RelativePointDisplacement
              << " | maximum: "
              << maximumRelativePointDisplacement
              << " | samples: "
              << relativePointDisplacements.size()
              << '\n';

    std::cout << "[LOCAL BA RELATIVE POINT COUNTS] >= 10%: "
              << relativeAbove10Percent
              << " | >= 25%: "
              << relativeAbove25Percent
              << " | >= 50%: "
              << relativeAbove50Percent
              << " | >= 100%: "
              << relativeAbove100Percent
              << '\n';

    const double maximumAcceptedRelativePointDisplacement = 0.50;
    const double maximumAcceptedBadRelativePointRatio = 0.25;
    const double badRelativePointRatio = 
        relativePointDisplacements.empty() ? 1.0 : static_cast<double>(relativeAbove50Percent) / static_cast<double>(relativePointDisplacements.size());

    const bool excessiveRelativePointContamination = badRelativePointRatio > maximumAcceptedBadRelativePointRatio;

    std::cout << "[LOCAL BA RELATIVE WINDOW] points >= 50%: "
              << relativeAbove50Percent
              << "/"
              << relativePointDisplacements.size()
              << " | ratio: "
              << badRelativePointRatio
              << " | allowed ratio: "
              << maximumAcceptedBadRelativePointRatio
              << '\n';

    if (excessiveRelativePointContamination) {
        std::cout << "[LOCAL BA WRITEBACK SKIP]"
                  << " relative point contamination"
                  << " | points >= 50%: "
                  << relativeAbove50Percent
                  << "/"
                  << relativePointDisplacements.size()
                  << " | ratio: "
                  << badRelativePointRatio
                  << " | allowed: "
                  << maximumAcceptedBadRelativePointRatio
                  << '\n';

        return true;
    }

    if (excessiveRotation || excessiveCurrentPoseCorrection) {
        std::cout << "[LOCAL BA WRITEBACK SKIP] max rotation correction: "
                  << maximumRotationCorrectionDegrees
                  << " deg"
                  << " | current center correction: "
                  << currentPoseCenterCorrection
                  << " | allowed center correction: "
                  << maximumAllowedPoseCenterCorrection
                  << '\n';

        return true;
    }

    if (pointDisplacements.empty()) {
        std::cerr << "Error: no finite map point available for BA write-back.\n";
        return false;
    }

    std::sort(pointDisplacements.begin(), pointDisplacements.end());

    const std::size_t p90Index = 
        static_cast<std::size_t>(0.90 * static_cast<double>(pointDisplacements.size() - 1));

    const double p90PointDisplacement = pointDisplacements[p90Index];

    const double maximumAcceptedPointDisplacement =
        std::max(1.0, 3.0 * p90PointDisplacement);

    // -------------------------------------------------------------------
    // 3. Réinjection des points 3D
    // 
    // Un point qui a énormément dérivé pendant le BA est conservé à son 
    // ancienne position.
    // -------------------------------------------------------------------
    std::size_t updatedPoints = 0;
    std::size_t rejectedPoints = 0;

    for (std::size_t localPointIndex = 0;
         localPointIndex < localProblem.points.size();
         ++localPointIndex) {

        const LocalBundleAdjustmentPoint& localPoint = localProblem.points[localPointIndex];

        const int globalIndex = localPoint.globalMapPointIndex;

        const cv::Point3f& before = mapPoints[globalIndex];
        const std::array<double, 3>& after = localPoint.position;

        const double dx = after[0] - static_cast<double>(before.x);
        const double dy = after[1] - static_cast<double>(before.y);
        const double dz = after[2] - static_cast<double>(before.z);

        const double displacement = std::sqrt(dx*dx + dy*dy + dz*dz);
        const double relativeDisplacement = relativePointDisplacementByPoint[localPointIndex];

        const bool invalidPointUpdate = 
            !std::isfinite(displacement) || !std::isfinite(relativeDisplacement);

        const bool excessiveAbsoluteDisplacement = displacement > maximumAcceptedPointDisplacement;
        const bool excessiveRelativeDisplacement = relativeDisplacement > maximumAcceptedRelativePointDisplacement;

        if (invalidPointUpdate || excessiveAbsoluteDisplacement || excessiveRelativeDisplacement) {
            // Exclusion permanente du BA uniquement si la correction est invalide ou relativement disproportionnée
            if (invalidPointUpdate || excessiveRelativeDisplacement) {
                excludedFromLocalBundleAdjustment.insert(localPoint.mapPointId);
            }

            // Dans tous les cas de rejet, on NE réinjecte PAS la position optimisée dans la carte
            ++rejectedPoints;
            continue;
        }

        mapPoints[globalIndex] =
            cv::Point3f(
                static_cast<float>(after[0]),
                static_cast<float>(after[1]),
                static_cast<float>(after[2])
            );

        ++updatedPoints;
    }

    // -------------------------------------------------------------------
    // 4. Réinjection des poses des keyframes
    // -------------------------------------------------------------------
    bool currentPoseUpdated = false;
    bool mappingAnchorUpdated = false;

    for (const LocalBundleAdjustmentPose& localPose : localProblem.poses) {
        cv::Mat rotationVector =
            (cv::Mat_<double>(3, 1)
                << localPose.parameters[0],
                   localPose.parameters[1],
                   localPose.parameters[2]);

        cv::Mat optimizedRotation;
        cv::Rodrigues(
            rotationVector, optimizedRotation
        );

        cv::Mat optimizedTranslation =
            (cv::Mat_<double>(3, 1)
                << localPose.parameters[3],
                   localPose.parameters[4],
                   localPose.parameters[5]);

        // ---------------------------------------------------------------
        // Mise à jour de la pose stockée dans l'historique
        // ---------------------------------------------------------------
        auto historyIterator =
            std::find_if(
                keyframeHistory.begin(),keyframeHistory.end(),
                [&localPose](const KeyframeRecord& record) {
                    return record.frameIndex == localPose.frameIndex;
                }
            );

        if (historyIterator == keyframeHistory.end()) {
            std::cerr << "Error: optimized keyframe not found in history"
                      << " | frame: "
                      << localPose.frameIndex
                      << '\n';

            return false;
        }

        historyIterator->worldToCameraRotation = optimizedRotation.clone();
        historyIterator->translationVector = optimizedTranslation.clone();

        // ----------------------------------------------------------------
        // La frame actuellement traitée peut elle-même avoir été optimisée
        // par le BA.
        // ----------------------------------------------------------------
        if (localPose.frameIndex == currentFrameIndex) {
            worldToCameraRotation = optimizedRotation.clone();
            translationVector = optimizedTranslation.clone();

            worldFromCameraRotation =
                worldToCameraRotation.t();

            cameraPositionWorld =
                -worldToCameraRotation.t() * translationVector;

            currentPoseUpdated = true;
        }


        // ----------------------------------------------------------------
        // L'ancre de triangulation peut également appartenir à la 
        // fenêtre locale.
        // ----------------------------------------------------------------
        if (localPose.frameIndex == mappingAnchorFrameIndex) {
            keyframeWorldToCameraRotation = optimizedRotation.clone();
            keyframeTranslationVector = optimizedTranslation.clone();


            mappingAnchorUpdated = true;
        }
    }

    std::cout << "[LOCAL BA WRITEBACK] updated points: "
              << updatedPoints
              << "/"
              << localProblem.points.size()
              << " | rejected points: "
              << rejectedPoints
              << " | threshold: "
              << maximumAcceptedPointDisplacement
              << " | current pose: "
              << (currentPoseUpdated ? "updated" : "not in window")
              << " | mapping anchor: "
              << (mappingAnchorUpdated ? "updated" : "not in window")
              << '\n';

    return true;
}

int main(int argc, char* argv[]) {
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
    // map_visual_odometry
    //     sequence_main.mp4
    //     camera_calibration_video_1920x1080.yaml
    //     trajectory.csv
    //     0.5
    // ------------------------------------------------------------
    if (argc < 4) {
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

    if (argc >= 5) {
        try {
            samplingIntervalSeconds = std::stod(argv[4]);
        } catch (...) {
            std::cerr << "Error: invalid sampling interval.\n";
            return 1;
        }
    }

    if (samplingIntervalSeconds <= 0.0) {
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


    if (!calibrationFile.isOpened()) {
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
        cameraMatrix.cols != 3) {
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


    if (!video.isOpened()) {
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


    if (fps <= 0.0) {
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
        videoHeight != calibrationHeight) {
        std::cerr << "Error: video resolution does not match "
                  << "the calibration resolution.\n";

        return 1;
    }

    // Conversion de l'intervalle temporel en nombre de frames
    // 
    // Exemple:
    //  20 fps * 0.5 s = 10 frames
    const int frameStep = std::max(
        1, static_cast<int>(std::round(fps * samplingIntervalSeconds))
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
    if (outputCsvPath.has_parent_path()) {
        fs::create_directories(outputCsvPath.parent_path());
    }

    std::ofstream trajectoryFile(outputCsvPath);

    if (!trajectoryFile.is_open()) {
        std::cerr << "Error: could not create trajectory file.\n";
        return 1;
    }

    trajectoryFile << "frame_index,"
                   << "time_seconds,"
                   << "x, "
                   << "y, "
                   << "z, "
                   << "good_matches,"
                   << "pose_inliers,"
                   << "is_keyframe,"
                   << "segment_id\n";


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


    // -------------------------------------------------------------
    // Critères minimaux pour accepter une pose PnP
    // 
    //  - au moins 20 correspondances 3D-2D avant PnP;
    //  - au moins 20 inliers après RANSAC;
    //  - au moins 40 % des correspondances doivent être des inliers.
    // 
    // Le ratio évite par exemple d'accepter une pose avec 20 inliers
    // sur 200 correspondances.
    // -------------------------------------------------------------
    const int minimumPnPCorrespondences = 20;
    //const int minimumPnPInliers = 20;
    const int minimumPnPInliers = 15;

    const double minimumPnPInlierRatio = 0.40;

    // Entre 15 & 19 inliers, on exigera une proportion nettement plus
    // forte de correspondances cohérentes
    const double minimumSparsePnPInlierRatio = 0.60;


    // ------------------------------------------------------------
    // Carte 3D persistante
    // 
    // Invariant important:
    // mapPoints[i] correspond toujours à mapDescriptors.row(i)
    // 
    // Chaque ligne de descripteur décrit donc le même point 
    // physique que la position 3D de même indice.
    // ------------------------------------------------------------
    std::vector<cv::Point3f> mapPoints;
    cv::Mat mapDescriptors;

    std::vector<MapPointQuality> mapPointQualities;


    // Identifiant stable de chaque point 3D.
    // Contrairement à l'indice dans mapPoints, cet ID ne change
    // jamais lors d'un culling.
    std::vector<MapPointId> mapPointIds;
    MapPointId nextMapPointId = 0;
    bool mapInitialized = false;

    // Points jugés instables par le write-back du BA
    //
    // On les conserve dans la carte pour le matching/PnP,
    // mais ils ne participeront plus aux prochains BA locaux.
    std::unordered_set<MapPointId> excludedFromLocalBundleAdjustment;


    // ------------------------------------------------------------
    // Keyframe courante
    // Une keyframe est une image de référence dont on conserve:
    //      - les keypoints ORB;
    //      - les descripteurs;
    //      - les associations keypoint -> point 3D de la carte;
    //      - sa pose absolue dans le repère monde.
    // 
    // keyframePointIndices[i] = -1
    //      -> le keypoint i n'est associé à aucun point 3D.
    // keyframePointIndices[i] >= 0
    //      -> indice du point correspondant dans mapPoints.
    // ------------------------------------------------------------
    std::vector<cv::KeyPoint> keyframeKeypoints;
    cv::Mat keyframeDescriptors;
    std::vector<int> keyframeMapPointIndices;

    // Pose monde -> caméra de la keyframe:
    //      X_camera = Rcw * X_world + tcw
    cv::Mat keyframeWorldToCameraRotation;
    cv::Mat keyframeTranslationVector;

    int keyframeFrameIndex = -1;
    
    // Dernière frame ayant déclenché la politique de sélection
    // de keyframe.
    // 
    // Elle est distincte de keyframeFrameIndex :
    // keyframeframeIndex désigne désormais l'ancre réellement 
    // utilisée pour la triangulation de nouveaux points 3D.
    int lastSelectedKeyframeFrameIndex = -1;

    // Une keyframe candidate ne devient nouvelle ancre de 
    // triangulation que si elle a permis d'enrichir suffisamment
    // la carte.

    // Une extension d'au moins 10 points est suffisamment utile pour
    // justifier une nouvelle ancre.
    const int minimumNewMapPointsForAnchorPromotion = 10;

    // Même si peu de nouveaux points ont été créés, on ne laisse pas
    // l'ancre devenir trop trop ancienne.
    // 
    // 4 * frameStep = environ 2 secondes
    const int maximumMappingAnchorGapFrames = 4 * frameStep;

    // Historique des keyframes sélectionnées.
    // 
    // Pour l'instant on les conserve toutes.
    // Le local BA utilisera ensuite uniquement les dernières
    // keyframes de cet historique.
    std::vector<KeyframeRecord> keyframeHistory;

    keyframeHistory.reserve(64);



    // ----------------------------------------------------------------
    // Politique adaptative de création des keyframes.
    // Minimum: 
    // On évite 2 keyframes trop proches, ce qui donnerait 
    // peu de parallaxe.
    // 
    // Maximum:
    // même si le tracking reste bon, on renouvelle périodiquement
    // la référence.
    // ----------------------------------------------------------------
    const int minimumKeyframeGapFrames = 2 * frameStep;  // environ 1 seconde
    const int maximumKeyframeGapFrames = 6 * frameStep;  // environ 3 seconde

    // Tracking considéré comme affaibli si l'un de ces critères devient mauvais
    const int keyframeInlierThreshold = 50;
    const double keyframeInlierRatioThreshold = 0.60;

    const std::size_t keyframeMatchThreshold = 60;
    
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
    cv::Mat worldFromCameraRotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat cameraPositionWorld = cv::Mat::zeros(3, 1, CV_64F);


    // ------------------------------------------------------------
    // Historique utilisé pour vérifier la cohérence du mouvement
    // 
    // lastAcceptedPoseFrameIndex : 
    //      frame correspondant à cameraPositionWorld
    // 
    // recentNormalizedStepDistances :
    //      derniers déplacements acceptés, ramenés à un intervalle 
    //      normal de frameStep
    // ------------------------------------------------------------
    int lastAcceptedPoseFrameIndex = -1;

    std::vector<double> recentNormalizedStepDistances;


    // Nombre de déplacements récents utilisés pour estimer 
    // le mouvement "normal"
    const std::size_t motionHistorySize = 5;

    // On autorise un deplacement jusqu'à 3 fois la médiane 
    // récente avant de le considérer comme suspect.
    const double maximumStepMedianMultiplier = 3.0;

    // Sécurité pour éviter qu'une période quasi immobile
    // rende le seuil exceesivement petit
    // 
    // ATTENTION: ce sont toujours des unités arbitraires
    // monoculaires, pas des mètres
    const double minimumMaximumNormalizedStep = 0.5;

    // ----------------------------------------------------------------------
    // Etat du tracking 
    // 
    // On considère le tracking comme perdu après plusieurs frames
    // sélectionées consécutives pour lesquelles la localisation par rapport
    // à la carte a échoué.
    // 
    // Pour  cette première étape A10, TRACKING LOST est seulement
    // diagnostique : aucune carte n'est encore supprimée ou recréée.
    // -----------------------------------------------------------------------
    int consecutiveTrackingFailures = 0;
    int longestTrackingFailureStreak = 0;
    int trackingLossEvents = 0;
    int recoveryGeometryAttempts = 0;
    int usableRecoveryGeometryCandidates = 0;
    int recovery3DValidationAttempts = 0;
    int validRecoveryBootstrapCandidates = 0;
    int recoveryConfirmationEvents = 0;
    int lastValidRecoveryBootstrapFrameIndex = -1;
    int recoveryRebootstrapEvents = 0;
    int trajectorySegmentId = 0;
    int stableRecoveryRebootstrapEvents = 0;
    int unstableRecoveryRebootstrapEvents = 0;
    int recoveryPostValidationSuccessfulPnP = 0;
    int recoveryPostValidationSegmentId = -1;

    bool recoveryPostValidationPending = false;
    bool recoveryBootstrapConfirmed = false;

    const int requiredSuccessfulPnPForStableRecovery = 2;
    const int maximumRecoveryConfirmationGapFrames = 4 * frameStep;

    // Référence locale utilisée uniquement pendant TRACKING LOST.
    //
    // Elle est indépendante de previousKeypoints/Descriptors,
    // qui restent associés à la dernière pose globale fiable.
    std::vector<cv::KeyPoint> recoveryReferenceKeypoints;
    cv::Mat recoveryReferenceDescriptors;

    int recoveryReferenceFrameIndex = -1;

    bool recoveryReferenceInitialized = false;
    bool trackingLost = false;

    const int trackingLossFailureThreshold = 4;

    auto registerTrackingFailure = [&](int failedFrameIndex, const char* reason) {

        ++consecutiveTrackingFailures;

        if (recoveryPostValidationPending && recoveryPostValidationSuccessfulPnP > 0) {
            recoveryPostValidationSuccessfulPnP = 0;

            std::cout << "[RECOVERY VALIDATION RESET] segment "
                      << recoveryPostValidationSegmentId
                      << " | frame: "
                      << failedFrameIndex
                      << " | reason: "
                      << reason
                      << '\n';
        }

        longestTrackingFailureStreak = std::max(longestTrackingFailureStreak, consecutiveTrackingFailures);

        std::cout << "[TRACKING FAILURE] frame "
                  << failedFrameIndex
                  << " | consecutive: "
                  << consecutiveTrackingFailures
                  << "/"
                  << trackingLossFailureThreshold
                  << " | reason: "
                  << reason
                  << '\n';

        if (!trackingLost && consecutiveTrackingFailures >= trackingLossFailureThreshold) {
            trackingLost = true;

            if (recoveryPostValidationPending) {
                ++unstableRecoveryRebootstrapEvents;

                std::cout << "[RECOVERY UNSTABLE] segment "
                          << recoveryPostValidationSegmentId
                          << " | tracking lost again at frame "
                          << failedFrameIndex
                          << '\n';


                recoveryPostValidationPending = false;
                recoveryPostValidationSuccessfulPnP = 0;
                recoveryPostValidationSegmentId = -1;
            }

            lastValidRecoveryBootstrapFrameIndex = -1;
            recoveryBootstrapConfirmed = false;

            ++trackingLossEvents;

            std::cout << "[TRACKING LOST] frame "
                      << failedFrameIndex
                      << " | consecutive failures: "
                      << consecutiveTrackingFailures
                      << '\n';
        }
    };

    auto registerTrackingSuccess = [&](int successfulFrameIndex) {
        if (trackingLost) {
            std::cout << "[TRACKING RECOVERED] frame "
                      << successfulFrameIndex
                      << " | previous failure streak: "
                      << consecutiveTrackingFailures
                      << '\n';
        }

        consecutiveTrackingFailures = 0;
        trackingLost = false;

        recoveryReferenceKeypoints.clear();
        recoveryReferenceDescriptors.release();

        lastValidRecoveryBootstrapFrameIndex = -1;
        recoveryBootstrapConfirmed = false;

        recoveryReferenceFrameIndex = -1;
        recoveryReferenceInitialized = false;
    };


    // ------------------------------------------------------------
    // Lecture séquentielle de la vidéo
    // ------------------------------------------------------------
    cv::Mat frame;

    int frameIndex = 0;
    int selectedFrames = 0;
    int successfulPoseSteps = 0;

    while (true) {
        const bool frameRead = video.read(frame);

        if (!frameRead) {
            std::cout
                << "\n[VIDEO] Reading stopped at frame "
                << frameIndex
                << " / "
                << frameCount
                << '\n';

            break;
        }

        // On ne traite qu'une frame sur frameStep.
        if (frameIndex % frameStep != 0) {
            ++frameIndex;
            continue;
        }

        ++selectedFrames;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);


        // Détection ORB sur la frame courante
        std::vector<cv::KeyPoint> currentKeypoints;
        cv::Mat currentDescriptors;

        orb->detectAndCompute(
            gray,
            cv::noArray(),
            currentKeypoints,
            currentDescriptors
        );

        if (currentDescriptors.empty()) {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : no descriptors\n";

            if (mapInitialized) {
                registerTrackingFailure(frameIndex, "no descriptors");
            }

            ++frameIndex;
            continue;
        }

        // Première frame: simple initialisation de la trajectoire
        if (!previousFrameInitialized) {
            previousGray = gray.clone();
            previousKeypoints = currentKeypoints;
            previousDescriptors = currentDescriptors.clone();
            previousFrameInitialized = true;

            trajectoryFile << frameIndex
                           << ","
                           << static_cast<double>(frameIndex) / fps
                           << ",0,0,0,0,0,0,"
                           << trajectorySegmentId
                           << '\n';

            std::cout << "[INIT] frame "
                      << frameIndex
                      << " : "
                      << currentKeypoints.size()
                      << " keypoints\n";

            ++frameIndex;
            continue;
        }


        auto tryRecoveryGeometry = [&](const char* triggerReason) {
            
            if (!mapInitialized || !trackingLost || currentDescriptors.empty()) {
                return;
            }

            // Première tentative après passage à TRACKING LOST:
            // la référence initiale est la dernière frame dont la pose globale était connue.
            if (!recoveryReferenceInitialized) {

                if (previousDescriptors.empty() || previousKeypoints.empty()) {
                    return;
                }

                recoveryReferenceKeypoints = previousKeypoints;
                recoveryReferenceDescriptors = previousDescriptors.clone();
                recoveryReferenceFrameIndex = lastAcceptedPoseFrameIndex;
                recoveryReferenceInitialized = true;

                std::cout << "[RECOVERY REFERENCE INIT] frame "
                          << recoveryReferenceFrameIndex
                          << '\n';
            }

            auto advanceRecoveryReference = [&]() {
                recoveryReferenceKeypoints = currentKeypoints;
                recoveryReferenceDescriptors = currentDescriptors.clone();
                recoveryReferenceFrameIndex = frameIndex;
            };

            ++recoveryGeometryAttempts;

            // Matching dernière pose connue -> frame courante
            std::vector<std::vector<cv::DMatch>> recoveryKnnMatches;

            matcher.knnMatch(
                recoveryReferenceDescriptors, currentDescriptors,
                recoveryKnnMatches, 2
            );

            std::vector<cv::DMatch> recoveryCandidateMatches;

            for (const auto& candidates : recoveryKnnMatches) {

                if (candidates.size() < 2) {
                    continue;
                }

                if (candidates[0].distance < ratioThreshold * candidates[1].distance) {
                    recoveryCandidateMatches.push_back(candidates[0]);
                }
            }

            // On privilégie les meilleurs matches avant de supprimer les duplications éventuelles côté image courante.
            std::sort(
                recoveryCandidateMatches.begin(), recoveryCandidateMatches.end(),
                [](const cv::DMatch& a, const cv::DMatch& b) {
                    return a.distance < b.distance; 
                }
            );

            std::vector<bool> recoveryCurrentKeypointAlreadyUsed(
                currentKeypoints.size(), false
            );

            std::vector<cv::DMatch> recoveryGoodMatches;
            recoveryGoodMatches.reserve(recoveryCandidateMatches.size());

            for (const cv::DMatch& match : recoveryCandidateMatches) {

                if (recoveryCurrentKeypointAlreadyUsed[match.trainIdx]) {
                    continue;
                }
                
                recoveryCurrentKeypointAlreadyUsed[match.trainIdx] = true;
                recoveryGoodMatches.push_back(match);
            }

            if (recoveryGoodMatches.size() < 20) {
                std::cout << "[RECOVERY GEOMETRY] reference: "
                          << recoveryReferenceFrameIndex
                          << " -> current: "
                          << frameIndex
                          << " | matches: "
                          << recoveryGoodMatches.size()
                          << " | usable: no"
                          << " | reason: insufficient matches"
                          << " | trigger: "
                          << triggerReason
                          << '\n';

                advanceRecoveryReference();
                return;
            }

            // Points correspondants
            std::vector<cv::Point2f> recoveryPreviousPoints;
            std::vector<cv::Point2f> recoveryCurrentPoints;

            recoveryPreviousPoints.reserve(recoveryGoodMatches.size());
            recoveryCurrentPoints.reserve(recoveryGoodMatches.size());

            for (const cv::DMatch& match : recoveryGoodMatches) {
                
                recoveryPreviousPoints.push_back(recoveryReferenceKeypoints[match.queryIdx].pt);
                recoveryCurrentPoints.push_back(currentKeypoints[match.trainIdx].pt);
            }

            // Undistortion
            std::vector<cv::Point2f> recoveryPreviousUndistorted;
            std::vector<cv::Point2f> recoveryCurrentUndistorted;

            cv::undistortPoints(
                recoveryPreviousPoints, recoveryPreviousUndistorted,
                cameraMatrix, distortionCoefficients,
                cv::noArray(), cameraMatrix
            );

            cv::undistortPoints(
                recoveryCurrentPoints, recoveryCurrentUndistorted,
                cameraMatrix, distortionCoefficients,
                cv::noArray(), cameraMatrix
            );

            // Matrice essentielle
            cv::Mat recoveryEssentialMask;

            const cv::Mat recoveryEssentialMatrix = 
                cv::findEssentialMat(
                    recoveryPreviousUndistorted, recoveryCurrentUndistorted,
                    cameraMatrix, cv::RANSAC,
                    0.999, 1.0, recoveryEssentialMask
                );

            if (recoveryEssentialMatrix.empty()) {
                std::cout << "[RECOVERY GEOMETRY] reference: "
                          << recoveryReferenceFrameIndex
                          << " -> current: "
                          << frameIndex
                          << " | matches: "
                          << recoveryGoodMatches.size()
                          << " | usable: no"
                          << " | reason: essential matrix failed"
                          << '\n';

                advanceRecoveryReference();
                return;
            }

            // Pose relative
            cv::Mat recoveryRotation;
            cv::Mat recoveryTranslation;

            cv::Mat recoveryPoseMask = recoveryEssentialMask.clone();

            const int recoveryPoseInliers = 
                cv::recoverPose(
                    recoveryEssentialMatrix,
                    recoveryPreviousUndistorted, recoveryCurrentUndistorted,
                    cameraMatrix,
                    recoveryRotation, recoveryTranslation,
                    recoveryPoseMask
                );

            const double recoveryInlierRatio = 
                static_cast<double>(recoveryPoseInliers) / static_cast<double>(recoveryGoodMatches.size());

            const bool recoveryGeometryUsable = recoveryPoseInliers >= 15 && recoveryInlierRatio >= 0.40;

            std::cout << "[RECOVERY GEOMETRY] reference: "
                      << recoveryReferenceFrameIndex
                      << " -> current: "
                      << frameIndex
                      << " | matches: "
                      << recoveryGoodMatches.size()
                      << " | pose inliers: "
                      << recoveryPoseInliers
                      << " | ratio: "
                      << recoveryInlierRatio
                      << " | usable: "
                      << (recoveryGeometryUsable ? "yes" : "no")
                      << " | trigger: "
                      << triggerReason
                      << '\n';

            if (!recoveryGeometryUsable) {
                advanceRecoveryReference();
                return;
            }

            ++usableRecoveryGeometryCandidates;
            ++recovery3DValidationAttempts;

            std::vector<cv::Point2f> recoveryInlierPreviousPoints;
            std::vector<cv::Point2f> recoveryInlierCurrentPoints;

            std::vector<cv::DMatch> recoveryInlierMatches;

            recoveryInlierPreviousPoints.reserve(recoveryPoseInliers);
            recoveryInlierCurrentPoints.reserve(recoveryPoseInliers);
            recoveryInlierMatches.reserve(recoveryPoseInliers);


            for (std::size_t i = 0; i < recoveryGoodMatches.size(); ++i) {
                if (recoveryPoseMask.at<uchar>(static_cast<int>(i)) == 0) {
                    continue;
                }

                recoveryInlierPreviousPoints.push_back(recoveryPreviousUndistorted[i]);
                recoveryInlierCurrentPoints.push_back(recoveryCurrentUndistorted[i]);
                recoveryInlierMatches.push_back(recoveryGoodMatches[i]);
            }

            // ------------------------------------------------------------
            // Validation 3D du candidat de re-bootstrap
            // ------------------------------------------------------------
            const double minimumRecoveryParallaxDegrees = 0.25;
            const double maximumRecoveryReprojectionError = 2.0;
            const int minimumRecoveryValid3DPoints = 30;
            const double minimumRecoveryValid3DRatio = 0.50;

            // Première caméra locale : [I | 0]
            cv::Mat recoveryFirstCameraPose = cv::Mat::zeros(3, 4, CV_64F);
            cv::Mat recoveryIdentity = cv::Mat::eye(3, 3, CV_64F);

            recoveryIdentity.copyTo(recoveryFirstCameraPose(cv::Rect(0, 0, 3, 3)));


            // Deuxième caméra locale : [R | t]
            cv::Mat recoverySecondCameraPose;

            cv::hconcat(
                recoveryRotation, recoveryTranslation,
                recoverySecondCameraPose
            );


            const cv::Mat recoveryProjection1 = cameraMatrix * recoveryFirstCameraPose;
            const cv::Mat recoveryProjection2 = cameraMatrix * recoverySecondCameraPose;


            // Triangulation
            cv::Mat recoveryHomogeneousPoints4D;

            cv::triangulatePoints(
                recoveryProjection1, recoveryProjection2,
                recoveryInlierPreviousPoints, recoveryInlierCurrentPoints,
                recoveryHomogeneousPoints4D
            );


            cv::Mat recoveryHomogeneousPoints4D64;
            recoveryHomogeneousPoints4D.convertTo(recoveryHomogeneousPoints4D64, CV_64F);


            // Centres caméra dans le repère local de la première caméra
            const cv::Mat recoveryCameraCenter1 = cv::Mat::zeros(3, 1, CV_64F);
            const cv::Mat recoveryCameraCenter2 = -recoveryRotation.t() * recoveryTranslation;


            // Validation des points 3D
            int recoveryValid3DPoints = 0;
            int recoveryLowParallaxRejected = 0;
            int recoveryReprojectionRejected = 0;

            std::vector<double> recoveryParallaxAngles;
            std::vector<cv::Point3f>recoveryBootstrapMapPoints;

            cv::Mat recoveryBootstrapMapDescriptors;

            std::vector<int> recoveryBootstrapReferenceMapPointIndices(recoveryReferenceKeypoints.size(), -1);
            std::vector<int> recoveryBootstrapCurrentMapPointIndices(currentKeypoints.size(), -1);

            recoveryBootstrapMapPoints.reserve(recoveryInlierMatches.size());

            for (int i = 0; i < recoveryHomogeneousPoints4D64.cols; ++i) {

                const double w = recoveryHomogeneousPoints4D64.at<double>(3, i);

                if (std::abs(w) < 1e-9) {
                    continue;
                }

                const double x = recoveryHomogeneousPoints4D64.at<double>(0, i) / w;
                const double y = recoveryHomogeneousPoints4D64.at<double>(1, i) / w;
                const double z = recoveryHomogeneousPoints4D64.at<double>(2, i) / w;


                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    continue;
                }


                cv::Mat pointCamera1 =
                    (cv::Mat_<double>(3, 1)
                        << x, y, z);

                cv::Mat pointCamera2 = recoveryRotation * pointCamera1 + recoveryTranslation;


                // Le point doit être devant les deux caméras.
                if (pointCamera1.at<double>(2, 0) <= 0.0 || pointCamera2.at<double>(2, 0) <= 0.0) {
                    continue;
                }


                // --------------------------------------------------------
                // Parallaxe
                // --------------------------------------------------------
                const cv::Mat ray1 = pointCamera1 - recoveryCameraCenter1;
                const cv::Mat ray2 = pointCamera1 - recoveryCameraCenter2;


                const double ray1Norm = cv::norm(ray1);
                const double ray2Norm = cv::norm(ray2);

                if (ray1Norm < 1e-9 || ray2Norm < 1e-9) {
                    continue;
                }


                double cosineAngle = ray1.dot(ray2) / (ray1Norm * ray2Norm);
                cosineAngle = std::clamp(cosineAngle, -1.0, 1.0);

                const double parallaxDegrees =std::acos(cosineAngle) * 180.0 / CV_PI;

                recoveryParallaxAngles.push_back(parallaxDegrees);

                if (parallaxDegrees < minimumRecoveryParallaxDegrees) {

                    ++recoveryLowParallaxRejected;
                    continue;
                }


                // Reprojection première caméra
                cv::Mat projected1 = cameraMatrix * pointCamera1;

                const cv::Point2d reprojection1(
                    projected1.at<double>(0, 0) / projected1.at<double>(2, 0),
                    projected1.at<double>(1, 0) / projected1.at<double>(2, 0)
                );


                // Reprojection deuxième caméra
                cv::Mat projected2 = cameraMatrix * pointCamera2;

                const cv::Point2d reprojection2(
                    projected2.at<double>(0, 0) / projected2.at<double>(2, 0),
                    projected2.at<double>(1, 0) / projected2.at<double>(2, 0)
                );

                const double reprojectionError1 = cv::norm(reprojection1 - cv::Point2d(recoveryInlierPreviousPoints[i]));
                const double reprojectionError2 = cv::norm(reprojection2 - cv::Point2d(recoveryInlierCurrentPoints[i]));


                if (reprojectionError1 > maximumRecoveryReprojectionError ||
                    reprojectionError2 > maximumRecoveryReprojectionError) {

                    ++recoveryReprojectionRejected;
                    continue;
                }

                const cv::DMatch& recoveryMatch = recoveryInlierMatches[i];
                const int recoveryMapPointIndex = static_cast<int>(recoveryBootstrapMapPoints.size());


                recoveryBootstrapMapPoints.emplace_back(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)
                );

                recoveryBootstrapMapDescriptors.push_back(currentDescriptors.row(recoveryMatch.trainIdx));

                recoveryBootstrapReferenceMapPointIndices[recoveryMatch.queryIdx] = recoveryMapPointIndex;
                recoveryBootstrapCurrentMapPointIndices[recoveryMatch.trainIdx] = recoveryMapPointIndex;

                ++recoveryValid3DPoints;
            }


            // Parallaxe médiane
            double recoveryMedianParallax = 0.0;

            if (!recoveryParallaxAngles.empty()) {

                std::sort(
                    recoveryParallaxAngles.begin(), recoveryParallaxAngles.end()
                );

                recoveryMedianParallax = recoveryParallaxAngles[recoveryParallaxAngles.size() / 2];
            }


            // Ratio de points 3D réellement valides
            const double recoveryValid3DRatio =
                recoveryInlierPreviousPoints.empty() ? 
                    0.0 : static_cast<double>(recoveryValid3DPoints) / static_cast<double>(recoveryInlierPreviousPoints.size());

            // Verdict final
            const bool validRecoveryBootstrapCandidate =
                recoveryValid3DPoints >= minimumRecoveryValid3DPoints &&
                recoveryValid3DRatio >= minimumRecoveryValid3DRatio;

            bool recoveryConfirmedThisFrame = false;

            if (validRecoveryBootstrapCandidate) {

                ++validRecoveryBootstrapCandidates;

                // Confirmation temporelle du re-bootstrap
                //
                // Un seul bon candidat ne suffit pas encore.
                // On demande un deuxième candidat 3D valide suffisamment
                // proche temporellement.
                if (!recoveryBootstrapConfirmed) {
                    if (lastValidRecoveryBootstrapFrameIndex >= 0) {

                        const int recoveryConfirmationGap = frameIndex - lastValidRecoveryBootstrapFrameIndex;

                        if (recoveryConfirmationGap <= maximumRecoveryConfirmationGapFrames) {

                            recoveryBootstrapConfirmed = true;
                            recoveryConfirmedThisFrame = true;
                            ++recoveryConfirmationEvents;

                            std::cout << "[RECOVERY CONFIRMED] frame "
                                      << frameIndex
                                      << " | previous candidate: "
                                      << lastValidRecoveryBootstrapFrameIndex
                                      << " | gap: "
                                      << recoveryConfirmationGap
                                      << " frames"
                                      << " | maximum allowed: "
                                      << maximumRecoveryConfirmationGapFrames
                                      << '\n';
                        }
                        else {
                            std::cout << "[RECOVERY CANDIDATE REARMED] frame "
                                      << frameIndex
                                      << " | previous candidate: "
                                      << lastValidRecoveryBootstrapFrameIndex
                                      << " | gap: "
                                      << recoveryConfirmationGap
                                      << " frames"
                                      << " | maximum allowed: "
                                      << maximumRecoveryConfirmationGapFrames
                                      << '\n';
                        }
                    }
                    else {
                        std::cout << "[RECOVERY CANDIDATE ARMED] frame "
                                  << frameIndex
                                  << '\n';
                    }


                    // Ce candidat devient la référence temporelle du prochain candidat valide.
                    lastValidRecoveryBootstrapFrameIndex = frameIndex;
                }
            }

            // Diagnostic
            std::cout << "[RECOVERY 3D] reference: "
                      << recoveryReferenceFrameIndex
                      << " -> current: "
                      << frameIndex
                      << " | pose inliers: "
                      << recoveryInlierPreviousPoints.size()
                      << " | valid 3D: "
                      << recoveryValid3DPoints
                      << " | valid ratio: "
                      << recoveryValid3DRatio
                      << " | median parallax: "
                      << recoveryMedianParallax
                      << " deg"
                      << " | low-parallax rejected: "
                      << recoveryLowParallaxRejected
                      << " | reprojection rejected: "
                      << recoveryReprojectionRejected
                      << " | REBOOT CANDIDATE: "
                      << (validRecoveryBootstrapCandidate ? "yes" : "no")
                      << '\n';


            if (recoveryConfirmedThisFrame) {

                // ---------------------------------------------------------------------------
                // Nouveau segment monoculaire
                //
                // La frame recoveryReferenceFrameIndex définit le nouveau repère local :
                //
                //     Rcw = I
                //     tcw = 0
                //
                // La frame courante possède la pose [recoveryRotation | recoveryTranslation].
                //
                // Il n'existe PAS de transformation métrique connue permettant
                // de recoller rigoureusement ce segment à l'ancien.
                // ---------------------------------------------------------------------------
                const int recoveryBootstrapReferenceFrameIndex = recoveryReferenceFrameIndex;

                ++trajectorySegmentId;
                ++recoveryRebootstrapEvents; 

                recoveryPostValidationPending = true;

                recoveryPostValidationSuccessfulPnP = 0;
                recoveryPostValidationSegmentId = trajectorySegmentId;

                // Remplacement de l'ancienne carte
                mapPoints = recoveryBootstrapMapPoints;
                mapDescriptors = recoveryBootstrapMapDescriptors.clone();


                mapPointQualities.clear();

                mapPointQualities.resize(mapPoints.size());


                for (MapPointQuality& quality : mapPointQualities) {
                    quality.creationFrameIndex = frameIndex;
                    quality.lastSeenFrameIndex = frameIndex;
                    quality.lastMatchedFrameIndex = frameIndex;
                }


                // Nouveaux IDs.
                //
                // nextMapPointId n'est PAS remis à zéro : les IDs restent uniques sur toute l'exécution.
                mapPointIds.clear();

                mapPointIds.reserve(mapPoints.size());


                for (std::size_t i = 0; i < mapPoints.size(); ++i) {
                    mapPointIds.push_back(nextMapPointId++);
                }


                if (mapPoints.size() != static_cast<std::size_t>(mapDescriptors.rows) ||
                    mapPoints.size() != mapPointQualities.size() || 
                    mapPoints.size() != mapPointIds.size()) {

                    std::cerr << "Error: recovery map invariant broken"
                              << " | points: "
                              << mapPoints.size()
                              << " | descriptors: "
                              << mapDescriptors.rows
                              << " | qualities: "
                              << mapPointQualities.size()
                              << " | IDs: "
                              << mapPointIds.size()
                              << '\n';

                    return;
                }


                // Les exclusions BA appartenaient à l'ancienne carte.
                excludedFromLocalBundleAdjustment.clear();

                // Le BA ne doit jamais mélanger deux segments ayant des repères / échelles monoculaires indépendants.
                keyframeHistory.clear();

                // Première keyframe du nouveau segment : frame de référence de la paire de récupération.
                const cv::Mat recoveryReferenceRotation = cv::Mat::eye(3, 3, CV_64F);

                const cv::Mat recoveryReferenceTranslation = cv::Mat::zeros(3, 1, CV_64F);


                KeyframeRecord recoveryReferenceRecord;

                if (!createKeyframeRecord(
                        recoveryBootstrapReferenceFrameIndex,
                        recoveryReferenceKeypoints,
                        recoveryBootstrapReferenceMapPointIndices,
                        mapPointIds,
                        recoveryReferenceRotation, recoveryReferenceTranslation,
                        recoveryReferenceRecord)) {

                    std::cerr << "Error: could not create recovery reference keyframe.\n";
                    return;
                }

                keyframeHistory.push_back(std::move(recoveryReferenceRecord));

                // Deuxième keyframe = frame actuelle
                KeyframeRecord recoveryCurrentRecord;

                if (!createKeyframeRecord(
                        frameIndex,
                        currentKeypoints,
                        recoveryBootstrapCurrentMapPointIndices,
                        mapPointIds,
                        recoveryRotation, recoveryTranslation,
                        recoveryCurrentRecord)) {

                    std::cerr << "Error: could not create recovery current keyframe.\n";

                    return;
                }


                keyframeHistory.push_back(std::move(recoveryCurrentRecord));


                // La frame courante devient l'ancre active.
                keyframeFrameIndex = frameIndex;

                lastSelectedKeyframeFrameIndex = frameIndex;

                keyframeKeypoints = currentKeypoints;

                keyframeDescriptors = currentDescriptors.clone();

                keyframeMapPointIndices = recoveryBootstrapCurrentMapPointIndices;

                keyframeWorldToCameraRotation = recoveryRotation.clone();
                keyframeTranslationVector = recoveryTranslation.clone();


                // État de pose du nouveau segment
                worldFromCameraRotation = recoveryRotation.t();
                cameraPositionWorld = -recoveryRotation.t() * recoveryTranslation;
                lastAcceptedPoseFrameIndex = frameIndex;


                // L'échelle du nouveau segment vient d'être réinitialisée.
                // L'historique de mouvement de l'ancien segment n'est donc
                // plus comparable.
                recentNormalizedStepDistances.clear();


                // La frame actuelle devient la nouvelle frame fiable.
                previousGray = gray.clone();
                previousKeypoints = currentKeypoints;
                previousDescriptors = currentDescriptors.clone();

                mapInitialized = true;

                ++successfulPoseSteps;


                // CSV : origine du nouveau segment.
                //
                // On écrit rétroactivement la frame de référence puisque
                // aucune pose valide n'avait été enregistrée pour elle.
                trajectoryFile << recoveryBootstrapReferenceFrameIndex
                               << ","
                               << static_cast<double>(recoveryBootstrapReferenceFrameIndex) / fps
                               << ",0,0,0,0,0,1,"
                               << trajectorySegmentId
                               << '\n';


                const double recoveryX = cameraPositionWorld.at<double>(0, 0);
                const double recoveryY = cameraPositionWorld.at<double>(1, 0);
                const double recoveryZ = cameraPositionWorld.at<double>(2, 0);


                trajectoryFile << frameIndex
                               << ","
                               << static_cast<double>(frameIndex) / fps
                               << ","
                               << recoveryX
                               << ","
                               << recoveryY
                               << ","
                               << recoveryZ
                               << ","
                               << recoveryGoodMatches.size()
                               << ","
                               << recoveryPoseInliers
                               << ",1,"
                               << trajectorySegmentId
                               << '\n';


                std::cout << "[RECOVERY REBOOT] segment "
                          << trajectorySegmentId
                          << " | reference frame: "
                          << recoveryBootstrapReferenceFrameIndex
                          << " | current frame: "
                          << frameIndex
                          << " | new map points: "
                          << mapPoints.size()
                          << '\n';

                std::cout << "[RECOVERY VALIDATION ARMED] segment "
                          << recoveryPostValidationSegmentId
                          << " | required consecutive PnP: "
                          << requiredSuccessfulPnPForStableRecovery
                          << '\n';
                        

                printLocalBundleAdjustmentConnectivity(keyframeHistory, mapPointIds, 5);


                // Ceci remet trackingLost à false et nettoie les buffers de récupération.
                registerTrackingSuccess(frameIndex);

                return;
            }

            // La frame courante devient la référence locale
            // de la prochaine tentative.
            advanceRecoveryReference();
        };

        
        // ---------------------------------------------------------------
        // Localisation par rapport à la carte 3D existante
        // ---------------------------------------------------------------
        if (mapInitialized) {
            // Matching :
            //
            // mapDescriptors       = descripteurs associés aux points 3D
            // currentDescriptors   = descripteurs de l'image actuelle
            //
            // queryIdx -> indice dans la carte
            // trainIdx -> indice dans currentKeypoints
            std::vector<std::vector<cv::DMatch>> mapKnnMatches;

            matcher.knnMatch(
                mapDescriptors,
                currentDescriptors,
                mapKnnMatches,
                2
            );


            // ---------------------------------------------------
            // Matches ayant passé le test de Lowe
            //
            // Ils ne sont pas encore nécessairement univoques 
            // côté image courante : plusieurs points 3D peuvent
            // avoir choisi le même keypoint 2D
            // ---------------------------------------------------
            std::vector<cv::DMatch> mapCandidateMatches;
            
            for (const auto& candidates : mapKnnMatches) {
                if (candidates.size() < 2) {
                    continue;
                }

                if (candidates[0].distance < ratioThreshold * candidates[1].distance) {
                    mapCandidateMatches.push_back(candidates[0]);
                }
            }

            std::sort(
                mapCandidateMatches.begin(),
                mapCandidateMatches.end(),
                [](const cv::DMatch& a,
                    const cv::DMatch& b) {
                    return a.distance < b.distance;
                }
            );

            std::vector<bool> currentKeypointAlreadyUsed(
                currentKeypoints.size(), false
            );

            std::vector<cv::DMatch> mapGoodMatches;

            mapGoodMatches.reserve(
                mapCandidateMatches.size()
            );

            for (const cv::DMatch& match : mapCandidateMatches) {
                // trainIdx correspond au keypoint de la frame courante
                if (currentKeypointAlreadyUsed[match.trainIdx]) {
                    continue;
                }

                currentKeypointAlreadyUsed[match.trainIdx] = true;

                mapGoodMatches.push_back(
                    match
                );
            }

            const std::size_t removedDuplicateImageMatches =
                mapCandidateMatches.size() - mapGoodMatches.size();

            std::cout << "[MAP MATCHING] frame "
                      << frameIndex
                      << " | Lowe matches: "
                      << mapCandidateMatches.size()
                      << " | unique 2D matches: "
                      << mapGoodMatches.size()
                      << " | duplicates removed: "
                      << removedDuplicateImageMatches
                      << '\n';

            // ---------------------------------------------------------
            // Dernière apparition visuelle des points de carte.
            // 
            // mapGoodMatches a déjà passé:
            //      - Lowe
            //      - l'unicité côté keypoint 2D.
            // 
            // On met cette info à jour AVANT PnP.
            // Ainsi, même si la pose est ensuite rejetée, on sait que
            // le descripteur du point est encore retrouvé dans l'image.
            // ---------------------------------------------------------
            for (const cv::DMatch& match : mapGoodMatches) {
                const int mapPointIndex = match.queryIdx;

                if (mapPointIndex < 0 ||
                    mapPointIndex >= static_cast<int>(
                        mapPointQualities.size()
                        )) {
                    std::cerr << "Error: invalid map point index while updating last match.\n";
                    return 1;
                }

                mapPointQualities[mapPointIndex].lastMatchedFrameIndex = frameIndex;
            }
            // Construction des correspondances:
            // point 3D de la carte <-> point 2D de l'image
            std::vector<cv::Point3f> pnpObjectPoints;
            std::vector<cv::Point2f> pnpImagePoints;

            pnpObjectPoints.reserve(
                mapGoodMatches.size()
            );
            pnpImagePoints.reserve(
                mapGoodMatches.size()
            );

            for (const cv::DMatch& match : mapGoodMatches) {
                pnpObjectPoints.push_back(
                    mapPoints[match.queryIdx]
                );

                pnpImagePoints.push_back(
                    currentKeypoints[match.trainIdx].pt
                );
            }


            // Cas où il y a pas assez de correspondances 3D-2D: 
            // impossible de localiser proprement la caméra.
            if (pnpObjectPoints.size() < minimumPnPCorrespondences) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : only "
                          << pnpObjectPoints.size()
                          << " 3D-2D matches\n";

                registerTrackingFailure(frameIndex, "insufficient 3D-2D matches");
                tryRecoveryGeometry("insufficient 3D-2D matches");

                ++frameIndex;
                continue;
            }

            // Estimation robuste de la pose
            cv::Mat rotationVector;
            cv::Mat translationVector;
            cv::Mat pnpInliers;

            const bool pnpSuccess = 
                cv::solvePnPRansac(
                    pnpObjectPoints, pnpImagePoints,
                    cameraMatrix, distortionCoefficients,
                    rotationVector, translationVector,
                    false, 100, 3.0, 0.99,
                    pnpInliers,
                    cv::SOLVEPNP_ITERATIVE
                );


            // ------------------------------------------------------
            // Validation de la pose PnP
            // ------------------------------------------------------

            // 1ère condition : solvePnPRansac doit avoir réussi
            if (!pnpSuccess) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : solvePnPRansac failed\n";

                registerTrackingFailure(frameIndex, "solvePnPransac failed");
                tryRecoveryGeometry("solvePnPransac failed");

                ++frameIndex;
                continue;
            }

            // Proportion des correspondances 3D-2D qui ont réellement 
            // été considérées comme cohérentes par RANSAC.
            // Exemple: 80 matches, 60 inliers -> ratio = 60/80 = 0.75
            const double pnpInlierRatio =
                static_cast<double>(pnpInliers.rows) / static_cast<double>(pnpObjectPoints.size());

            // 2ème condition : quantité absolue suffisante d'inliers

            // Avec les correspondances 2D désormais univoques, 15 à 19 
            // inliers peuvent être suffisants si leur ratio est 
            // particulièrement bon
            if (pnpInliers.rows < minimumPnPInliers) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : "
                          << pnpInliers.rows
                          << " PnP inliers"
                          << " | ratio: "
                          << pnpInlierRatio
                          << '\n';

                registerTrackingFailure(frameIndex, "insufficient PnP inliers");
                tryRecoveryGeometry("insufficient PnP inliers");
                ++frameIndex;
                continue;
            }

            // Cas "sparse but clean": 
            // entre 15 & 19 inliers, on demande au moins 60%
            if (pnpInliers.rows < 20 && pnpInlierRatio < minimumSparsePnPInlierRatio) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : sparse PnP"
                          << " | inliers: "
                          << pnpInliers.rows
                          << " | ratio: "
                          << pnpInlierRatio
                          << '\n';

                registerTrackingFailure(frameIndex, "sparse PnP rejected");
                tryRecoveryGeometry("sparse PnP rejected");
                ++frameIndex;
                continue;
            }

            // 3ème condition : proportion suffisante d'inliers.
            // Une pose peut avoir 20 ou 30 inliers mais rester douteuse
            // si elle avait plusieurs centaines de correspondances
            // candidates au départ.
            if (pnpInlierRatio < minimumPnPInlierRatio) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : low inlier ratio"
                          << " | matches: "
                          << pnpObjectPoints.size()
                          << " | inliers: "
                          << pnpInliers.rows
                          << " | ratio: "
                          << pnpInlierRatio
                          << '\n';

                registerTrackingFailure(frameIndex, "low PnP inlier ratio");
                tryRecoveryGeometry("low PnP inlier ratio");
                ++frameIndex;
                continue;
            }


            
            // solvePnP donne: X_camera = Rcw * X_world + tcw
            // Position caméra dans le monde: Cw = -Rcw^T * tcw
            cv::Mat worldToCameraRotation;
            cv::Rodrigues(
                rotationVector,
                worldToCameraRotation
            );

            // --------------------------------------------------------
            // Association des keypoints de la frame courante avec les 
            // points 3D réellement validés par PnP/RANSAC.
            // --------------------------------------------------------
            std::vector<int> currentMapPointIndices(
                currentKeypoints.size(), -1
            );

            for (int i = 0; i < pnpInliers.rows; ++i) {
                // solvePnPRansac retourne les indices dans pnpObjectPoints /
                // pnpImagePoints.
                // Ces deux vecteurs ont exactement le même ordre que
                // mapGoodMatches.
                const int matchListIndex = pnpInliers.at<int>(i, 0);

                const cv::DMatch& inlierMatch = mapGoodMatches[matchListIndex];

                // queryIdx = point de carte
                // trainIdx = keypoint image courante
                currentMapPointIndices[inlierMatch.trainIdx] = inlierMatch.queryIdx;
            }

            const cv::Mat candidateCameraPositionWorld = 
                -worldToCameraRotation.t() * translationVector;

            // -----------------------------------------------------------
            // Cohérence temporelle de la nouvelle pose
            // -----------------------------------------------------------
            const int frameGap = 
                frameIndex - lastAcceptedPoseFrameIndex;

            // Distance 3D entre la dernière pose acceptée et la nouvelle
            // pose proposée par PnP
            const double rawStepDistance = 
                cv::norm(candidateCameraPositionWorld - cameraPositionWorld);

            // Si une ou plusieurs frames ont été rejetées,
            // rawStepDistance correspond à un temps plus long.
            //
            // On ramène donc toujours le déplacement à un intervalle
            // standard de frameStep.
            //
            // Exemple :
            //     frameStep = 15
            //
            // 15 frames écoulées : facteur 1
            // 30 frames écoulées : facteur 0.5
            // 45 frames écoulées : facteur 1/3
            const double normalizedStepDistance = 
                rawStepDistance * static_cast<double>(frameStep) / static_cast<double>(frameGap);
                

            double recentMedianStep = 0.0;
            double maximumAllowedStep = std::numeric_limits<double>::infinity();

            if (recentNormalizedStepDistances.size() >= motionHistorySize) {
                std::vector<double> sortedRecentSteps = recentNormalizedStepDistances;

                std::sort(sortedRecentSteps.begin(), sortedRecentSteps.end());

                recentMedianStep = sortedRecentSteps[sortedRecentSteps.size() / 2];

                maximumAllowedStep =
                    std::max(minimumMaximumNormalizedStep, maximumStepMedianMultiplier * recentMedianStep);
            }

            if (recentNormalizedStepDistances.size() >= motionHistorySize && normalizedStepDistance > maximumAllowedStep) {
                std::cout << "[PNP SKIP] frame "
                          << frameIndex
                          << " : motion jump"
                          << " | step: "
                          << normalizedStepDistance
                          << " | recent median: "
                          << recentMedianStep
                          << " | allowed: "
                          << maximumAllowedStep
                          << '\n';

                registerTrackingFailure(frameIndex, "motion consistency rejection");
                tryRecoveryGeometry("motion consistency rejection");
                ++frameIndex;
                continue;
            }

            // La pose PnP a maintenant passé toutes les validations
            registerTrackingSuccess(frameIndex);

            const cv::Mat cameraPositionWorldBeforeAcceptedStep = cameraPositionWorld.clone();

            cameraPositionWorld = candidateCameraPositionWorld.clone();
            worldFromCameraRotation = worldToCameraRotation.t();

            lastAcceptedPoseFrameIndex = frameIndex;
            
            // Tous les matches envoyés à PnP comptent comme tentative
            // d'observation du point 3D
            for (const cv::DMatch& match : mapGoodMatches) {
                const int mapPointIndex = match.queryIdx;

                ++mapPointQualities[mapPointIndex].pnpMatchCount;
            }

            // Les inliers PnP sont des observations confirmées
            for (int i = 0; i < pnpInliers.rows; ++i) {
                const int matchListIndex = pnpInliers.at<int>(i, 0);

                const cv::DMatch& inlierMatch = mapGoodMatches[matchListIndex];

                const int mapPointIndex = inlierMatch.queryIdx;

                MapPointQuality& quality = mapPointQualities[mapPointIndex];

                ++quality.pnpInlierCount;

                quality.lastSeenFrameIndex = frameIndex;

                // Mise à jour du descripteur du point 3D.
                //
                // La correspondance vient d'être validée comme inlier PnP,
                // donc le descripteur observé dans la frame courante constitue
                // une nouvelle apparence fiable de ce point.
                //
                // Cela permet à la carte de suivre progressivement les
                // changements de point de vue.
                currentDescriptors.row(inlierMatch.trainIdx).copyTo(mapDescriptors.row(mapPointIndex));
            }

            ++successfulPoseSteps;

            if (recoveryPostValidationPending) {
                ++recoveryPostValidationSuccessfulPnP;

                std::cout << "[RECOVERY VALIDATION PNP] segment "
                    << recoveryPostValidationSegmentId
                    << " | frame: "
                    << frameIndex
                    << " | consecutive successful PnP: "
                    << recoveryPostValidationSuccessfulPnP
                    << "/"
                    << requiredSuccessfulPnPForStableRecovery
                    << '\n';

                if (recoveryPostValidationSuccessfulPnP >= requiredSuccessfulPnPForStableRecovery) {
                    ++stableRecoveryRebootstrapEvents;

                    std::cout << "[RECOVERY STABLE] segment "
                        << recoveryPostValidationSegmentId
                        << " | confirmed at frame "
                        << frameIndex
                        << " | consecutive successful PnP: "
                        << recoveryPostValidationSuccessfulPnP
                        << '\n';


                    recoveryPostValidationPending = false;
                    recoveryPostValidationSuccessfulPnP = 0;
                    recoveryPostValidationSegmentId = -1;
                }
            }

            // -----------------------------------------------------------
            // Décision adaptative de création d'une nouvelle keyframe
            // -----------------------------------------------------------
            const int framesSinceLastKeyframe = frameIndex - lastSelectedKeyframeFrameIndex;

            // Trop proche de la dernière keyframe: on préfère attendre davantage de baseline
            const bool enoughFrameSeparation = framesSinceLastKeyframe >= minimumKeyframeGapFrames;

            // Même avec un bon tracking, on ne conserve pas une keyframe indéfiniment
            const bool maximumGapReached = framesSinceLastKeyframe >= maximumKeyframeGapFrames;

            // ---------------------------------------------------------
            // Qualité actuelle du tracking
            // ---------------------------------------------------------

            // Peu de correspondances carte -> image
            const bool lowMapVisibility =
                mapGoodMatches.size() < keyframeMatchThreshold;

            // Peu d'inliers géométriquement cohérents
            const bool lowAbsolutePnPSupport =
                pnpInliers.rows < keyframeInlierThreshold;

            // Une proportion importante des matches est rejetée par RANSAC
            const bool lowPnPInlierRatio =
                pnpInlierRatio < keyframeInlierRatioThreshold;

            const bool trackingIsWeakening =
                lowMapVisibility ||
                lowAbsolutePnPSupport ||
                lowPnPInlierRatio;

            // -------------------------------------------------------
            // Création de la keyframe
            // Possibilités:
            //      1. le tracking commence à faiblir ET suffisamment
            //          de temps s'est écoulé;
            //      2. le délai maximal est atteint, même si le 
            //          tracking reste encore bon. 
            // -------------------------------------------------------
            const bool shouldCreateKeyframe = 
                maximumGapReached || (enoughFrameSeparation && trackingIsWeakening);

            if (shouldCreateKeyframe) {
                std::cout << "[KEYFRAME] frame "
                          << frameIndex
                          << " selected"
                          << " | reason: "
                          << (maximumGapReached ? "maximum gap" : "tracking weakening")
                          << " | gap: "
                          << framesSinceLastKeyframe
                          << " frames"
                          << " | matches: "
                          << mapGoodMatches.size()
                          << " | PnP inliers: "
                          << pnpInliers.rows
                          << " | ratio: "
                          << pnpInlierRatio
                          << " | previous selected: "
                          << lastSelectedKeyframeFrameIndex
                          << " | mapping anchor: "
                          << keyframeFrameIndex
                          << '\n';

                // --------------------------------------------------------
                // Extension de la carte
                // 
                // IMPORTANT :
                // ceci doit être fait AVANT de remplacer l'ancienne
                // keyframe par la frame courante.
                // --------------------------------------------------------
                const std::size_t mapSizeBefore = mapPoints.size();

                const int addedMapPoints = 
                    addNewMapPointsBetweenKeyframes(
                        keyframeKeypoints,
                        keyframeDescriptors,
                        keyframeMapPointIndices,
                        keyframeWorldToCameraRotation,
                        keyframeTranslationVector,

                        currentKeypoints,
                        currentDescriptors,
                        currentMapPointIndices,
                        worldToCameraRotation,
                        translationVector,

                        cameraMatrix,
                        distortionCoefficients,

                        ratioThreshold,

                        mapPoints,
                        mapDescriptors
                    );
                
                std::cout << "[MAP] Added "
                          << addedMapPoints
                          << " new points"
                          << " | size: "
                          << mapSizeBefore
                          << " -> "
                          << mapPoints.size()
                          << '\n';
                
                // -------------------------------------------------------
                // Création des infos de qualité pour les nouveaux points
                // 3D ajoutés pendant cette extension de carte.
                // 
                // Avant addNewMapPointsBetweenKeyframes():
                //      mapSizeBefore = ancien nombre de points.
                // 
                // Après:
                //      les indices [mapSizeBefore, mapPoints.size()]
                //      correspondent exactement aux nouveaux points.
                // -----------------------------------------------------
                for (std::size_t mapPointIndex = mapSizeBefore;
                    mapPointIndex < mapPoints.size();
                    ++mapPointIndex) {
                    MapPointQuality quality;

                    quality.creationFrameIndex = frameIndex;
                    quality.lastSeenFrameIndex = frameIndex;
                    quality.lastMatchedFrameIndex = frameIndex;

                    mapPointQualities.push_back(
                        quality
                    );

                    mapPointIds.push_back(
                        nextMapPointId++
                    );
                }

                // Verif permanente de notre invariant
                if (mapPoints.size() != 
                        static_cast<std::size_t>(mapDescriptors.rows) || 
                    mapPoints.size() != 
                        mapPointQualities.size() ||
                    mapPoints.size() != 
                        mapPointIds.size()
                   ) {
                    std::cerr << "Error: map data size mismatch"
                              << " | points: "
                              << mapPoints.size()
                              << " | descriptors: "
                              << mapDescriptors.rows
                              << " | qualities: "
                              << mapPointQualities.size()
                              << " | IDs: "
                              << mapPointIds.size()
                              << '\n';

                    return 1;
                }


                // -----------------------------------------------------------
                // Diagnostic de qualité de la carte
                // Aucun point n'est supprimé pour l'instant
                // -----------------------------------------------------------
                int neverConfirmedByPnP = 0;
                int confirmedAtLeastOnce = 0;
                int confirmedAtLeastThreeTimes = 0;
                int staleMapPoints = 0;

                int oldNeverConfirmed = 0;
                int lowPnPConfirmationRatio = 0;

                const int stalePointAgeFrames =
                    12 * frameStep;                               // environ 6 secondes

                for (const MapPointQuality& quality : mapPointQualities) {
                    if (quality.pnpInlierCount == 0) {
                        ++neverConfirmedByPnP;
                    }

                    if (quality.pnpInlierCount >= 1) {
                        ++confirmedAtLeastOnce;
                    }

                    if (quality.pnpInlierCount >= 3) {
                        ++confirmedAtLeastThreeTimes;
                    }

                    const int framesSinceLastSeen =
                        frameIndex - quality.lastSeenFrameIndex;

                    if (framesSinceLastSeen >= stalePointAgeFrames) {
                        ++staleMapPoints;
                    }

                    const int pointAgeFrames =
                        frameIndex - quality.creationFrameIndex;

                    // Point suffisamment ancien pour avoir eu plusieurs occasions
                    // d'être réobservé, mais jamais validé par PnP.
                    if (pointAgeFrames >= 12 * frameStep &&
                        quality.pnpInlierCount == 0) {
                        ++oldNeverConfirmed;
                    }

                    // Point qui a souvent produit un match, mais qui est rarement
                    // accepté comme inlier PnP.
                    // On exige au moins 5 tentatives pour ne pas tirer de 
                    // conclusion à partir de trop peu d'observations.
                    if (quality.pnpMatchCount >= 5) {
                        const double confirmationRatio = 
                            static_cast<double>(quality.pnpInlierCount) /
                            static_cast<double>(quality.pnpMatchCount);

                        if (confirmationRatio < 0.25) {
                            ++lowPnPConfirmationRatio;
                        }
                    }
                }

                std::cout << "[MAP QUALITY]"
                          << " size: "
                          << mapPoints.size()
                          << " | never PnP-confirmed: "
                          << neverConfirmedByPnP
                          << " | confirmed >= 1: "
                          << confirmedAtLeastOnce
                          << " | confirmed >= 3: "
                          << confirmedAtLeastThreeTimes
                          << " | stale: "
                          << staleMapPoints
                          << " | old never-confirmed: "
                          << oldNeverConfirmed
                          << " | low confirmation ratio: "
                          << lowPnPConfirmationRatio
                          << '\n';


                const int minimumCullingAgeFrames = 12 * frameStep;

                // Un point qui a encore produit un bon match au cours des
                // 4 derniers intervalles (~2 secondes) est protégé du culling
                const int maximumUnmatchedAgeFrames = 4 * frameStep;

                const int culledMapPoints = 
                    cullOldNeverConfirmedMapPoints(
                        frameIndex, minimumCullingAgeFrames, maximumUnmatchedAgeFrames,
                        mapPoints, mapDescriptors, mapPointQualities, mapPointIds,
                        keyframeMapPointIndices, currentMapPointIndices
                    );

                if (culledMapPoints < 0) {
                    return 1;
                }

                if (mapPoints.size() != static_cast<std::size_t>(mapDescriptors.rows) ||
                    mapPoints.size() != mapPointQualities.size() ||
                    mapPoints.size() != mapPointIds.size()) {
                    std::cerr << "Error: map data size mismatch after culling"
                              << " | points: "
                              << mapPoints.size()
                              << " | descriptors: "
                              << mapDescriptors.rows
                              << " | qualities: "
                              << mapPointQualities.size()
                              << " | IDs: "
                              << mapPointIds.size()
                              << '\n';

                    return 1;
                }


                // ------------------------------------------------------------------
                // La triangulation a potentiellement ajouté de nouvelles
                // associations au mapping anchor.
                // 
                // Son ancien KeyframeRecord a été créé avant que ces nouveaux points
                // actuelles, après remapping du culling.
                // ------------------------------------------------------------------
                if (!refreshStoredKeyframeRecord(
                    keyframeHistory,

                    keyframeFrameIndex,

                    keyframeKeypoints,
                    keyframeMapPointIndices,

                    mapPointIds,

                    keyframeWorldToCameraRotation,
                    keyframeTranslationVector)) {
                    return 1;
                }

                // ----------------------------------------------------------------
                // Sauvegarde de la keyframe dans l'hstorique destiné au 
                // Bundle Adjustement.
                // 
                // Le culling vient d'être effectué:
                // currentMapPointIndices utilise donc déjà les indices compactés
                // de la carte.
                // 
                // createKeyframeRecord() convertit immédiatement ces indices
                // temporaires en MapPointId permanents.
                // ----------------------------------------------------------------
                KeyframeRecord currentKeyframeRecord;

                if (!createKeyframeRecord(
                    frameIndex,

                    currentKeypoints,
                    currentMapPointIndices,

                    mapPointIds,

                    worldToCameraRotation,
                    translationVector,

                    currentKeyframeRecord)) {
                    return 1;
                }

                keyframeHistory.push_back(
                    std::move(currentKeyframeRecord)
                );

                

                std::cout << "[KEYFRAME HISTORY] frame: "
                          << frameIndex
                          << " | observations: "
                          << keyframeHistory.back().observations.size()
                          << " | stored keyframes: "
                          << keyframeHistory.size()
                          << '\n';


                printLocalBundleAdjustmentConnectivity(
                    keyframeHistory,
                    mapPointIds,
                    5
                );


                if (keyframeHistory.size() >= 5) {
                    LocalBundleAdjustmentProblem localBundleAdjustmentProblem;

                    if (!buildLocalBundleAdjustmentProblem(
                        keyframeHistory,
                        mapPoints, mapPointIds,
                        excludedFromLocalBundleAdjustment,
                        cameraMatrix,distortionCoefficients,
                        5,
                        localBundleAdjustmentProblem)) {

                        return 1;
                    }

                    if (!runLocalBundleAdjustment(
                        localBundleAdjustmentProblem, cameraMatrix)) {

                        std::cerr << "Error: local bundle adjustment failed.\n";
                        return 1;
                    }

                    if (!applyLocalBundleAdjustmentResults(
                            localBundleAdjustmentProblem,
                            mapPoints, mapPointIds,
                            excludedFromLocalBundleAdjustment,
                            keyframeHistory,
                            frameIndex, worldToCameraRotation, translationVector, worldFromCameraRotation, cameraPositionWorld,
                            keyframeFrameIndex, keyframeWorldToCameraRotation, keyframeTranslationVector,
                            std::max(
                                1.0, maximumAllowedStep
                            ))) {

                        std::cerr << "Error: local bundle adjustment write-back failed.\n";
                        return 1;
                    }
                    
                }
                
                // -------------------------------------------------------------
                // Sauvegarde 
                // -------------------------------------------------------------
                
                // La frame courante devient maintenant la nouvelle keyframe
                lastSelectedKeyframeFrameIndex = frameIndex;


                const int framesSinceMappingAnchor = 
                    frameIndex - keyframeFrameIndex;

                const bool enoughNewMapPoints = 
                    addedMapPoints >= minimumNewMapPointsForAnchorPromotion;

                const bool mappingAnchorTooOld = 
                    framesSinceMappingAnchor >= maximumMappingAnchorGapFrames;


                // -----------------------------------------------------------
                // Promotion éventuelle de la frame courante comme nouvelle
                // ancre de triangulation.
                // 
                // La sélection de la keyframe et le changement d'ancre sont
                // désormais deux décisions différentes.
                // -----------------------------------------------------------
                if (enoughNewMapPoints || mappingAnchorTooOld) {
                    keyframeFrameIndex = frameIndex;

                    keyframeKeypoints = currentKeypoints;
                    keyframeDescriptors = currentDescriptors.clone();

                    keyframeMapPointIndices = currentMapPointIndices;

                    keyframeWorldToCameraRotation = worldToCameraRotation.clone();
                    keyframeTranslationVector = translationVector.clone();

                    std::cout << "[MAP ANCHOR] promoted frame: "
                              << frameIndex
                              << " | reason: "
                              << (enoughNewMapPoints ? "map growth" : "anchor age")
                              << " | added points: "
                              << addedMapPoints
                              << " | previous anchor age: "
                              << framesSinceMappingAnchor
                              << " frames\n";
                } else {
                    std::cout << "[MAP ANCHOR] kept frame: "
                              << keyframeFrameIndex
                              << " | candidate frame: "
                              << frameIndex
                              << " | added only: "
                              << addedMapPoints
                              << " | anchor age: "
                              << framesSinceMappingAnchor
                              << " frames\n";
                }
            }

            // -----------------------------------------------------------
            // Mise à jour de l'historique dmouvement avec la pose 
            // FINALE réellement conservée.
            // 
            // Si un BA a été exécuté & accepté, cameraPositionWorld
            // contient ici la pose corrigée par le BA.
            // Sinon elle contient simplement la pose PnP.
            // -----------------------------------------------------------
            const double finalRawStepDistance =
                cv::norm(
                    cameraPositionWorld - cameraPositionWorldBeforeAcceptedStep
                );

            const double finalNormalizedStepDistance = 
                finalRawStepDistance * static_cast<double>(frameStep) / static_cast<double>(frameGap);

            recentNormalizedStepDistances.push_back(
                finalNormalizedStepDistance
            );

            if (recentNormalizedStepDistances.size() > motionHistorySize) {
                recentNormalizedStepDistances.erase(
                    recentNormalizedStepDistances.begin()
                );
            }


            const double x = cameraPositionWorld.at<double>(0, 0);
            const double y = cameraPositionWorld.at<double>(1, 0);
            const double z = cameraPositionWorld.at<double>(2, 0);

            // Sauvegarde CSV
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
                           << mapGoodMatches.size()
                           << ","
                           << pnpInliers.rows
                           << ","
                           << (shouldCreateKeyframe ? 1 : 0)
                           << ","
                           << trajectorySegmentId
                           << '\n';

            std::cout << "[PNP OK] frame "
                      << frameIndex
                      << " | map matches: "
                      << mapGoodMatches.size()
                      << " | PnP inliers: "
                      << pnpInliers.rows
                      << " | ratio: "
                      << pnpInlierRatio
                      << " | position: ["
                      << x
                      << ","
                      << y
                      << ","
                      << z
                      << "]\n";


            // Uitle pour la suite (pour l'extension dynamique de carte)
            previousGray = gray.clone();
            previousKeypoints = currentKeypoints;
            previousDescriptors = currentDescriptors.clone();

            ++frameIndex;

            // Très important: une fois PnP exécuté, on ne doit PAS 
            // continuer dans le pipeline Essentiel Matrix situé plus bas.
            continue;
        }

        // ---------------------------------------------------------------
        // Matching entre la dernière frame acceptée et la frame courante
        // ---------------------------------------------------------------
        std::vector<std::vector<cv::DMatch>> knnMatches;

        matcher.knnMatch(
            previousDescriptors, currentDescriptors, knnMatches, 2
        );


        std::vector<cv::DMatch> goodMatches;

        for (const auto& candidates : knnMatches) {
            if (candidates.size() < 2) {
                continue;
            }

            if (candidates[0].distance < ratioThreshold * candidates[1].distance) {
                goodMatches.push_back(candidates[0]);
            }
        }


        // On demande un minimum de correspondances pour éviter 
        // une estimation géométrique extrêmement fragile.
        if (goodMatches.size() < 20) {
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


        for (const cv::DMatch& match : goodMatches) {
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

        if (essentialMatrix.empty()) {
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

        if (poseInliers < 15) {
            std::cout << "[SKIP] frame "
                      << frameIndex
                      << " : only "
                      << poseInliers
                      << " pose inliers\n";

            ++frameIndex;
            continue;
        }


        // -------------------------------------------------------------
        // Initialisation de la carte
        // 
        // Ce bloc ne doit être exécuté qu'une et une seule fois.
        // -------------------------------------------------------------
        if (!mapInitialized) {
            // Correspondances qui ont ont réellement survécu à recoverPose
            std::vector<cv::Point2f> bootstrapPoints1;
            std::vector<cv::Point2f> bootstrapPoints2;

            std::vector<cv::DMatch> bootstrapMatches;

            bootstrapPoints1.reserve(
                poseInliers
            );

            bootstrapPoints2.reserve(
                poseInliers
            );

            bootstrapMatches.reserve(
                poseInliers
            );


            for (std::size_t i = 0; i < goodMatches.size(); ++i) {
                if (poseMask.at<uchar>(static_cast<int>(i)) == 0) {
                    continue;
                }

                bootstrapPoints1.push_back(
                    undistortedPrevious[i]
                );

                bootstrapPoints2.push_back(
                    undistortedCurrent[i]
                );

                bootstrapMatches.push_back(
                    goodMatches[i]
                );
            }

            // Première caméra: P0 = K [ I | 0 ]
            cv::Mat firstCameraPose = 
                cv::Mat::zeros(
                    3, 
                    4,
                    CV_64F
                );

            cv::Mat identity = 
                cv::Mat::eye(
                    3, 3, CV_64F
                );
                
            identity.copyTo(
                firstCameraPose(
                    cv::Rect(0, 0, 3, 3)
                    )
                );


            // Deuxième caméra: P1 = K [ R | t ]
            cv::Mat secondCameraPose;

            cv::hconcat(
                relativeRotation,
                relativeTranslation,
                secondCameraPose
            );

            const cv::Mat projectionMatrix1 = cameraMatrix * firstCameraPose;
            const cv::Mat projectionMatrix2 = cameraMatrix * secondCameraPose;


            // Triangulation
            cv::Mat homogeneousPoints4D;

            cv::triangulatePoints(
                projectionMatrix1,
                projectionMatrix2,
                bootstrapPoints1,
                bootstrapPoints2,
                homogeneousPoints4D
            );

            cv::Mat homogeneousPoints4D64;

            homogeneousPoints4D.convertTo(
                homogeneousPoints4D64,
                CV_64F
            );

            // Nettoyage de tout éventuelle donnée précédente.
            // Normalement inutile au premier passage, mais explicite
            mapPoints.clear();
            mapDescriptors.release();
            
            // Pour chaque keypoint de la deuxième image du bootstrap,
            // mémorisation éventuelle du point 3D correspondant.
            std::vector<int> bootstrapMapPointIndices(
                currentKeypoints.size(),
                -1
            );

            // Conversion en points 3D euclidiens + cheirality
            for (int i = 0; i < homogeneousPoints4D64.cols; ++i) {
                const double w = homogeneousPoints4D64.at<double>(
                    3, i
                );

                if (std::abs(w) < 1e-9) {
                    continue;
                };

                const double pointX = homogeneousPoints4D64.at<double>(0, i) / w;
                const double pointY = homogeneousPoints4D64.at<double>(1, i) / w;
                const double pointZ = homogeneousPoints4D64.at<double>(2, i) / w;

                if (!std::isfinite(pointX) || !std::isfinite(pointY) || !std::isfinite(pointZ)) {
                    continue;
                }

                cv::Mat pointCamera1 = 
                    (cv::Mat_<double>(3, 1)
                        << pointX, pointY, pointZ);

                cv::Mat pointCamera2 =
                    relativeRotation * pointCamera1 + relativeTranslation;

                const double depth1 = pointZ;
                const double depth2 = pointCamera2.at<double>(2, 0);

                if (depth1 <= 0.0 || depth2 <= 0.0) {
                    continue;
                }

                // Correspondance qui a produit ce point 3D
                const cv::DMatch& match = bootstrapMatches[i];

                // Indice que le nouveau point va recevoir dans mapPoints
                const int newMapPointIndex = static_cast<int>(mapPoints.size());

                mapPoints.emplace_back(
                    static_cast<float>(pointX),
                    static_cast<float>(pointY),
                    static_cast<float>(pointZ)
                );

                mapDescriptors.push_back(
                    currentDescriptors.row(match.trainIdx)
                );

                // Le keypoint de la deuxième image sait maintenant 
                // à quel point 3D il correspond.
                bootstrapMapPointIndices[match.trainIdx] = newMapPointIndex;
            }

            // Vérification de l'invariant carte/descripteurs
            if (mapPoints.size() != static_cast<std::size_t>(mapDescriptors.rows)) {
                std::cerr << "Error: map point/descriptor mismatch.\n";
                return 1;
            }

            if (mapPoints.size() < 20) {
                std::cout << "[MAP INIT FAILED] only "
                          << mapPoints.size()
                          << " valid 3D points\n";

                mapPoints.clear();
                mapDescriptors.release();

                ++frameIndex;
                continue;
            }

            // -----------------------------------------------------------
            // Initialisation des informations de qualité des points créés
            // pendant le bootstrap
            // 
            // On ne compte pas encore leurs deux observations de 
            // triangulation comme observations PnP.
            // 
            // pnpMatchCount / pnpInlierCount commencent donc à 0.
            // -----------------------------------------------------------
            mapPointQualities.clear();

            mapPointQualities.resize(
                mapPoints.size()
            );

            for (MapPointQuality& quality : mapPointQualities) {
                quality.creationFrameIndex = frameIndex;
                quality.lastSeenFrameIndex = frameIndex;
                quality.lastMatchedFrameIndex = frameIndex;
            }

            mapPointIds.clear();

            mapPointIds.reserve(
                mapPoints.size()
            );

            for (std::size_t i = 0; i < mapPoints.size(); ++i) {
                mapPointIds.push_back(
                    nextMapPointId++
                );
            }


            if (mapPointQualities.size() != mapPoints.size() ||
                mapPointIds.size() != mapPoints.size()) {
                std::cerr << "Error: map data mismatch after bootstrap"
                          << " | points: "
                          << mapPoints.size()
                          << " | qualities: "
                          << mapPointQualities.size()
                          << " | IDs: "
                          << mapPointIds.size()
                          << "\n";
                return 1;
            }

            mapInitialized = true;
            
            // --------------------------------------------------------
            // La deuxième image du bootstrap devient notre première
            // keyframe.
            // 
            // Sa pose monde -> caméra est exactement la pose relative
            // obtenue depuis la première caméra, puisque cette 
            // dernière définit le repère monde.
            // --------------------------------------------------------
            keyframeFrameIndex = frameIndex;
            keyframeKeypoints = currentKeypoints;
            keyframeDescriptors = currentDescriptors.clone();
            keyframeMapPointIndices = bootstrapMapPointIndices;
            keyframeWorldToCameraRotation = relativeRotation.clone();
            keyframeTranslationVector = relativeTranslation.clone();

            lastSelectedKeyframeFrameIndex = frameIndex;

            KeyframeRecord initialKeyframeRecord;

            if (!createKeyframeRecord(
                frameIndex,

                currentKeypoints,
                bootstrapMapPointIndices,

                mapPointIds,

                relativeRotation,
                relativeTranslation,

                initialKeyframeRecord)) {
                return 1;
            }


            keyframeHistory.push_back(
                std::move(initialKeyframeRecord)
            );

            std::cout << "[KEYFRAME HISTORY]"
                      << " frame: "
                      << frameIndex
                      << " | observations: "
                      << keyframeHistory.back().observations.size()
                      << " | stored keyframes: "
                      << keyframeHistory.size()
                      << '\n';

            printLocalBundleAdjustmentConnectivity(
                keyframeHistory,
                mapPointIds,
                5
            );

            std::cout << "\n[KEYFRAME] Initial keyframe: frame "
                      << keyframeFrameIndex
                      << '\n';

            std::cout << "\n=== Map initialization ===\n";

            std::cout << "Bootstrap frame: "
                      << frameIndex
                      << '\n';

            std::cout << "Pose inliers: "
                      << poseInliers
                      << '\n';

            std::cout << "Initial 3D map points: "
                      << mapPoints.size()
                      << '\n';

            std::cout << "Map descriptors: "
                      << mapDescriptors.rows
                      << " x "
                      << mapDescriptors.cols
                      << "\n\n";
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

        lastAcceptedPoseFrameIndex =
            frameIndex;

        ++successfulPoseSteps;


        // -------------------------------------------------------------
        // Sauvegarde de la trajectoire
        // -------------------------------------------------------------
        const double x = cameraPositionWorld.at<double>(0, 0);
        const double y = cameraPositionWorld.at<double>(1, 0);
        const double z = cameraPositionWorld.at<double>(2, 0);

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
                       << ","
                       << (mapInitialized ? 1 : 0)
                       << ","
                       << trajectorySegmentId
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

    std::cout << "Tracking loss events: "
              << trackingLossEvents
              << '\n';

    std::cout << "Longest tracking failure streak: "
              << longestTrackingFailureStreak
              << '\n';

    std::cout << "Recovery geometry attempts: "
              << recoveryGeometryAttempts
              << '\n';

    std::cout << "Usable recovery geometry candidates: "
              << usableRecoveryGeometryCandidates
              << '\n';

    std::cout << "Recovery 3D validation attempts: "
              << recovery3DValidationAttempts
              << '\n';

    std::cout << "Valid recovery bootstrap candidates: "
              << validRecoveryBootstrapCandidates
              << '\n';

    std::cout << "Recovery confirmation events: "
              << recoveryConfirmationEvents
              << '\n';

    std::cout << "Recovery re-bootstrap events: "
              << recoveryRebootstrapEvents
              << '\n';

    std::cout << "Stable recovery re-bootstrap events: "
              << stableRecoveryRebootstrapEvents
              << '\n';

    std::cout << "Unstable recovery re-bootstrap events: "
              << unstableRecoveryRebootstrapEvents
              << '\n';

    std::cout << "Recovery post-validation pending at end: "
              << (recoveryPostValidationPending ? "yes" : "no")
              << '\n';

    std::cout << "Trajectory segments: "
              << trajectorySegmentId + 1
              << '\n';

    std::cout << "Recovery bootstrap confirmed at end: "
              << (recoveryBootstrapConfirmed ? "yes" : "no")
              << '\n';


    std::cout << "Final tracking state: "
              << (trackingLost ? "LOST" : "TRACKING")
              << '\n';

    std::cout << "Trajectory saved to: "
                << outputCsvPath.string()
                << '\n';

    std::cout << "Last frame index reached: "
              << frameIndex
              << '\n';


    return 0;
}

