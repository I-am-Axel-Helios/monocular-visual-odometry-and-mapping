#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cctype>

#include <opencv2/opencv.hpp>


// Alias pour raccourcir std::filesystem
namespace fs = std::filesystem;


int main(int argc, char* argv[]) {
    // --------------------------------------------------------------------
    // Vérification des arguments
    //
    // Le programme recevra le dossier contenant les photos de calibration.
    //
    // Exemple : camera_calibration data/calibration
    // --------------------------------------------------------------------
    if (argc < 3) {
        std::cerr << "Usage: camera_calibration "
                  << "<calibration_directory> "
                  << "<output_calibration_file>\n";

        return 1;
    }

    const fs::path calibrationDirectory = argv[1];
    const fs::path outputPath = argv[2];

    if (!fs::exists(calibrationDirectory)) {
        std::cerr << "Error: directory does not exist: "
                  << calibrationDirectory
                  << '\n';

        return 1;
    }


    // ------------------------------------------------------------
    // Géométrie du damier
    //
    // Le damier contient :
    //     10 x 7 cases
    //
    // OpenCV attend le nombre de COINS INTERNES :
    //     9 x 6
    // ------------------------------------------------------------
    const cv::Size boardSize(9,6);


    // Pour connaître K, pas besoin forcément de la taille réelle des 
    // carrés. 
    //
    // En prenant le côté d'un carré comme unité, la calibration peut 
    // se faire en "largeurs de carré".
    const float squareSize = 23.0f;  // Valeur réelle: 2.3 cm = 23 mm



    // ------------------------------------------------------------
    // Construction du modèle 3D théorique de la mire
    //
    // La mire étant plane, on a : Z = 0
    // 
    // Les points ressemblent à :
    //
    // (0,0,0) (1,0,0) (2,0,0) ...
    // (0,1,0) (1,1,0) (2,1,0) ...
    // ------------------------------------------------------------
    std::vector<cv::Point3f> boardObjectPoints;

    for (int row = 0; row < boardSize.height; ++row) {
        for (int col = 0; col < boardSize.width; ++col) {
            boardObjectPoints.emplace_back(
                col * squareSize,
                row * squareSize,
                0.0f
            );
        }
    }


    // ------------------------------------------------------------
    // Recherche de toutes les images du dossier
    // ------------------------------------------------------------
    std::vector<fs::path> imageFiles;

    for (const auto& entry : fs::directory_iterator(calibrationDirectory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();

        // Transformation de l'extension en minuscule (pour pouvoir 
        // accepter .JPG, .JPEG et autres)
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );

        if (extension == ".jpg" ||
            extension == ".jpeg" ||
            extension == ".png") {
            imageFiles.push_back(entry.path());
        }
    }

    // Tri des fichiers par nom pour avoir une sortie prévisible
    std::sort(
        imageFiles.begin(),
        imageFiles.end()
    );

    std::cout << "Images found: "
        << imageFiles.size()
        << "\n\n";

    if (imageFiles.empty()) {
        std::cerr << "Error: no calibration images found.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // Détermination automatique de la résolution paysage dominante
    // 
    // (Vu que calibrateCamera() suppose que toutes les observations
    // appartiennent au même modèle de caméra et au même repère image,
    // on doit ignorer les portraits et les résolutions minoritaires
    // des images de la mire de calibration dans notre cas)
    // ------------------------------------------------------------

    // clé = (largeur, hauteur)
    // valeur = nombre d'images ayant cette résolution

    std::map<std::pair<int, int>, int> resolutionCounts;

    int portraitCount = 0;

    for (const fs::path& imagePath : imageFiles) {
        cv::Mat image = cv::imread(
            imagePath.string(),
            cv::IMREAD_COLOR
        );

        if (image.empty()) {
            continue;
        }

        if (image.rows > image.cols) {
            ++portraitCount;
            continue;
        }

        const std::pair<int, int> resolution(
            image.cols,
            image.rows
        );

        ++resolutionCounts[resolution];
    }

    if (resolutionCounts.empty()) {
        std::cerr << "Error: no landscape calibration images found.\n";
        return 1;
    }

    // Recherche de la résolution la plus représentée
    auto dominantResolution =
        std::max_element(
            resolutionCounts.begin(),
            resolutionCounts.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            }
        );

    const cv::Size calibrationImageSize(
        dominantResolution->first.first,
        dominantResolution->first.second
    );


    std::cout << "Portrait images ignored: "
        << portraitCount
        << '\n';


    std::cout << "Selected calibration resolution: "
        << calibrationImageSize.width
        << " x "
        << calibrationImageSize.height
        << '\n';


    std::cout << "Images with this resolution: "
        << dominantResolution->second
        << "\n\n";

    // ------------------------------------------------------------
    // Structures utilisées pour la calibration
    // 
    // objectPoints : positions 3D connues des 54 points
    // imagePoints : positions 2D detectées dans chaque photo
    // ------------------------------------------------------------
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Nous conservons aussi les noms pour pouvoir associer ensuite 
    // une erreur de rreprojection à chaque image.
    std::vector<std::string> acceptedImageNames;


    // Dossier de diagnostic, pour sauvegarder les images avec les coins dessinés
    // 
    // Exemple: outputPath = config/camera_calibration_1080x810.yaml
    //
    // outputPath.stem() va donc donner: camera_calibration_1080x810
    //
    // Les visualisations seront donc enregistrées dans: 
    // results/calibration/camera_calibration_photo_1080x810/detections/
    const fs::path calibrationResultDirectory =
        fs::path("results") / "calibration" / outputPath.stem();

    const fs::path detectionOutputDirectory =
        calibrationResultDirectory / "detections";

    fs::create_directories(
        detectionOutputDirectory
    );

    // ------------------------------------------------------------
    // Détection du damier dans chaque image
    // ------------------------------------------------------------
    for (const fs::path& imagePath : imageFiles) {
        cv::Mat image = cv::imread(
            imagePath.string(),
            cv::IMREAD_COLOR
        );

        if (image.empty()) {
            std::cout << "[LOAD FAIL] "
                << imagePath.filename().string()
                << '\n';

            continue;
        }

        // On n'utilisera que les photos paysage
        if (image.rows > image.cols) {
            std::cout << "[PORTRAIT]  "
                << imagePath.filename().string()
                << " -> ignored\n";

            continue;
        }

        // Toutes les images doivent avoir exactement la même
        // résolution que celle sélectionnée précédemment.
        if (image.size() != calibrationImageSize) {
            std::cout << "[SIZE SKIP] "
                << imagePath.filename().string()
                << " : "
                << image.cols
                << " x "
                << image.rows
                << '\n';

            continue;
        }



        // Conversion en niveaux de gris (pas besoin de la couleur pour les coins)
        cv::Mat gray;
        cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY
        );

        // Recherche des coins internes de la mire
        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(
            gray,
            boardSize,
            corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (!found) {
            std::cout << "[NOT FOUND] "
                << imagePath.filename().string()
                << '\n';

            continue;
        }


        // Raffinement subpixel
        //
        // La position initiale d'un coin est déjà bonne,
        // mais cornerSubPix améliore la précision à une fraction
        // de pixel.
        //
        // Cette précision est importante car nous essayons ensuite
        // d'estimer les paramètres intrinsèques à partir de ces
        // observations.
        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS |
                cv::TermCriteria::MAX_ITER,
                30,
                0.001
            )
        );


        // Ajout d'une nouvelle vue valide
        //
        // Le modèle 3D de la mire est identique à chaque fois.
        // Les coordonnées 2D observées changent avec le point
        // de vue de la caméra
        objectPoints.push_back(
            boardObjectPoints
        );

        imagePoints.push_back(
            corners
        );

        acceptedImageNames.push_back(
            imagePath.filename().string()
        );

        std::cout << "[OK]      "
            << imagePath.filename().string()
            << " : "
            << corners.size()
            << " corners\n";


        // Création d'une image de contrôle.
        //
        // Les coins trouvés sont dessinés afin que nous puissions
        // vérifier visuellement que la détection est correcte.
        cv::Mat detectionVisualization = image.clone();

        cv::drawChessboardCorners(
            detectionVisualization,
            boardSize,
            corners,
            found
        );


        const fs::path outputImagePath =
            detectionOutputDirectory /
            (
                imagePath.stem().string() + "_corners.jpg"
                );

        cv::imwrite(
            outputImagePath.string(),
            detectionVisualization
        );
    }


    const int acceptedImages = static_cast<int>(imagePoints.size());

    std::cout << "\nAccepted calibration images: "
        << acceptedImages
        << '\n';

    // Moins de 5 vues, c'est troop peu pour une calibration solide.
    // Donc on visera plutôt plus de 10 vues.
    if (acceptedImages < 5) {
        std::cerr << "Error: not enough valid calibration images.\n";
        return 1;
    }


    // -----------------------------------------------------------
    // Calibration
    // -----------------------------------------------------------

    // Matrice intrinsèque :
    // 
    //     [ fx   0   cx ]
    // K = [ 0   fy   cy ]
    //     [ 0    0    1 ]
    //
    cv::Mat cameraMatrix;



    // Coefficients de distorsion :
    // [ k1 k2 p1 p2 k3 ]
    // avec: 
    // - k1, k2, k3 : distorsion radiale
    // - p1, p2 : distorsion tangentielle
    //
    cv::Mat distortionCoefficients;


    // Pour chaque photo de calibration, OpenCV calcule également
    // la pose de la mire par rapport à la caméra.
    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;

    // Meilleure approximation possible (faire sauter le coefficient radial k3)
    const int calibrationFlags = cv::CALIB_FIX_K3;

    // Pour voir si annuler k3 et la distorsion tangentielle reussissent à 
    // expliquer quasiment toutes les observations (spoiler : non, la RMS et le
    // K driftent trop loin par rapport au modèle avec k3=0 seul)
    // const int calibrationFlags = cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST;

    const double calibrationRms =
        cv::calibrateCamera(
            objectPoints,
            imagePoints,
            calibrationImageSize,
            cameraMatrix,
            distortionCoefficients,
            rotationVectors,
            translationVectors,
            calibrationFlags
        );


    // ----------------------------------------------------------
    // Calcul de l'erreur de reprojection par image
    // 
    // Etape importante pour contrôler la qualité.
    //
    // À partir de K, de la distorsion et de la pose estimée,
    // on reprojette les points 3D de la mire vers l'image.
    //
    // On compare ensuite :
    // point réellement détecté
    //          VS
    // point prédit par la calibration
    // ----------------------------------------------------------
    std::vector<double> perViewErrors;

    double totalSquaredError = 0.0;
    std::size_t totalPoints = 0;


    std::cout << "\n=== Per-view reprojection errors ===\n";

    for (std::size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projectedPoints;

        cv::projectPoints(
            objectPoints[i],
            rotationVectors[i],
            translationVectors[i],
            cameraMatrix,
            distortionCoefficients,
            projectedPoints
        );


        // norm() donne ici sqrt(sum(error_i^2))
        const double l2Error =
            cv::norm(
                imagePoints[i],
                projectedPoints,
                cv::NORM_L2
            );


        // RMS pour cette vue: sqrt(sum(error_i^2) / N)
        const double viewRms =
            l2Error /
            std::sqrt(
                static_cast<double>(
                    objectPoints[i].size()
                    )
            );

        perViewErrors.push_back(viewRms);

        totalSquaredError += l2Error * l2Error;

        totalPoints += objectPoints[i].size();

        std::cout << acceptedImageNames[i]
            << " : "
            << std::fixed
            << std::setprecision(4)
            << viewRms
            << " px\n";
    }


    const double overallReprojectionError =
        std::sqrt(
            totalSquaredError / static_cast<double>(totalPoints)
        );

    // --------------------------------------------------
    // Résultats finaux
    // --------------------------------------------------
    std::cout << "\n=== Calibration result ===\n\n";

    std::cout << "Camera matrix K:\n"
        << cameraMatrix
        << "\n\n";

    std::cout << "Distortion coefficients:\n"
        << distortionCoefficients
        << "\n\n";

    std::cout << "OpenCV calibration RMS:\n"
        << calibrationRms
        << " px\n";

    std::cout << "Computed reprojection RMS:\n"
        << overallReprojectionError
        << " px\n";


    // -------------------------------------------------------
    // Sauvegarde de la calibration
    // 
    // outputPath est fourni par l'utilisateur au lancement du
    // programme. Cela permet au même outil de produire 
    // plusieurs calibrations sans écraser les précédentes.
    // --------------------------------------------------------

    // Crée automatiquement le dossier parent s'il n'existe pas
    if (outputPath.has_parent_path()) {
        fs::create_directories(
            outputPath.parent_path()
        );
    }

    cv::FileStorage file(
        outputPath.string(),
        cv::FileStorage::WRITE
    );

    

    if (!file.isOpened()) {
        std::cerr << "Error: could not create "
            << outputPath
            << '\n';

        return 1;
    }

    file
        << "image_width"
        << calibrationImageSize.width;

    file
        << "image_height"
        << calibrationImageSize.height;

    file
        << "board_width"
        << boardSize.width;

    file
        << "board_height"
        << boardSize.height;

    file
        << "square_size"
        << squareSize;

    file
        << "square_size_unit"
        << "mm";

    file
        << "accepted_images"
        << acceptedImages;

    file
        << "camera_matrix"
        << cameraMatrix;

    file
        << "distortion_coefficients"
        << distortionCoefficients;

    file
        << "opencv_calibration_rms"
        << calibrationRms;

    file
        << "reprojection_rms"
        << overallReprojectionError;

    file
        << "per_view_errors"
        << perViewErrors;

    file.release();


    std::cout << "\nCalibration saved to: "
        << outputPath.string()
        << '\n';

    std::cout << "Calibration diagnostics saved to: "
        << calibrationResultDirectory.string()
        << '\n';

    return 0;
}