#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;


// Une position de caméra enregistrée dans le CSV
struct TrajectoryPoint
{
	int frameIndex = 0;
	double timeSeconds = 0.0;

	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
};


std::string getTrajectoryDisplayName(
	const fs::path& csvPath
)
{
	const std::string filename = csvPath.stem().string();

	if (filename.find("map_based") != std::string::npos)
	{
		return "Map_based monocular VO";
	}

	if (filename.find("pairwise") != std::string::npos)
	{
		return "Pairwise monocular VO";
	}


	// Cas générique si on ajoute d'autres types de trajectoires
	std::string name = filename;

	std::replace(
		name.begin(),
		name.end(),
		'_',
		' '
	);

	return name;
}


int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(
		cv::utils::logging::LOG_LEVEL_ERROR
	);

	if (argc < 3)
	{
		std::cerr << "Usage: visualize_trajectory "
				  << "<trajectory_csv> <output_image>\n";
		return 1;
	}

	const fs::path inputCsvPath = argv[1];
	const fs::path outputImagePath = argv[2];

	const std::string trajectoryDisplayName = 
		getTrajectoryDisplayName(
			inputCsvPath
		);


	// ------------------------------------------------------------
	// Ouverture du fichier CSV
	// ------------------------------------------------------------
	std::ifstream file(inputCsvPath);

	if (!file.is_open())
	{
		std::cerr << "Error: could not open trajectory file: "
				  << inputCsvPath
				  << '\n';

		return 1;
	}


	std::vector<TrajectoryPoint> trajectory;

	// La 1ère ligne contient: 
	// frame_index, time_seconds, x, y, z ...
	// Donc on ne cherchera pas à la convertir en nombres.
	std::string line;
	std::getline(file, line);

	// ------------------------------------------------------------
	// Lecture des positions
	// ------------------------------------------------------------
	while (std::getline(file, line))
	{
		if (line.empty())
		{
			continue;
		}

		std::stringstream lineStream(line);

		std::vector<std::string> fields;
		std::string field;

		while (std::getline(
			lineStream,
			field,
			','
		))
		{
			fields.push_back(field);
		}


		// Notre CSV contient normalement:
		// frmae, time, x,  y, z, good_matches, pose_inliers
		if (fields.size() < 5)
		{
			continue;
		}


		try
		{
			TrajectoryPoint point;

			point.frameIndex = std::stoi(fields[0]);
			point.timeSeconds = std::stod(fields[1]);
			point.x = std::stod(fields[2]);
			point.y = std::stod(fields[3]);
			point.z = std::stod(fields[4]);

			trajectory.push_back(point);
		}
		catch (...)
		{
			std::cerr << "Warning: invalide CSV line ignored:\n"
					  << line
					  << '\n';
		}
	}


	file.close();


	if (trajectory.size() < 2)
	{
		std::cerr << "Error: not enough trajectory points.\n";
		return 1;
	}

	std::cout << "Trajectory points loaded: "
			  << trajectory.size()
			  <<  '\n';

	// ------------------------------------------------------------
	// Bornes de la trajectoire dans le plan X-Z
	// 
	// Pour une visualisation du dessus, on utilise:
	//	- horizontal = X
	//	- vertical = Z
	// 
	// Y sera analysé séparément comme dérive verticale
	// ------------------------------------------------------------
	double minX = trajectory.front().x;
	double maxX = trajectory.front().x;

	double minY = trajectory.front().y;
	double maxY = trajectory.front().y;

	double minZ = trajectory.front().z;
	double maxZ = trajectory.front().z;


	for (const TrajectoryPoint& point : trajectory)
	{
		minX = std::min(minX, point.x);
		maxX = std::max(maxX, point.x);

		minY = std::min(minY, point.y);
		maxY = std::max(maxY, point.y);

		minZ = std::min(minZ, point.z);
		maxZ = std::max(maxZ, point.z);
	}


	// ------------------------------------------------------------
	// Quelques stats
	// ------------------------------------------------------------
	double pathLength = 0.0;

	for (std::size_t i = 1; i < trajectory.size(); ++i)
	{
		const double dx = trajectory[i].x - trajectory[i - 1].x;
		const double dy = trajectory[i].y - trajectory[i - 1].y;
		const double dz = trajectory[i].z - trajectory[i - 1].z;

		pathLength += std::sqrt(dx*dx + dy*dy + dz*dz);
	}

	const double endDx = trajectory.back().x - trajectory.front().x;
	const double endDy = trajectory.back().y - trajectory.front().y;
	const double endDz = trajectory.back().z - trajectory.front().z;

	const double endDisplacement = 
		std::sqrt(
			endDx * endDx +
			endDy * endDy +
			endDz * endDz
		);

	std::cout << "Accumulated path length: "
			  << pathLength
			  << " arbitrary units\n";

	std::cout << "Start-to-end displacement: "
			  << endDisplacement
			  << " arbitrary units\n";

	std::cout << "Y range: "
			  << minY
			  << " -> "
			  << maxY
			  << '\n';

	// ------------------------------------------------------------
	// Création de l'image
	// ------------------------------------------------------------
	const int imageWidth = 1000;
	const int imageHeight = 800;

	const int margin = 70;


	cv::Mat visualization(
		imageHeight,
		imageWidth,
		CV_8UC3,
		cv::Scalar(255, 255, 255)
	);


	double rangeX = maxX - minX;
	double rangeZ = maxZ - minZ;


	// Evite une division par zéro si la trajectoire est quasiment 
	// parfaitement droite sur un axe
	if (rangeX < 1e-9)
	{
		rangeX = 1.0;
	}

	if (rangeZ < 1e-9)
	{
		rangeZ = 1.0;
	}


	const double scaleX =
		static_cast<double>(imageWidth - 2 * margin) / rangeX;

	const double scaleZ =
		static_cast<double>(imageHeight - 2 * margin) / rangeZ;

	// Même échelle sur X et Z :
	// on évite de déformer artificiellement la trajectoire.
	const double scale = std::min(scaleX, scaleZ);

	const double drawnWidth = rangeX * scale;
	const double drawnHeight = rangeZ * scale;

	const double offsetX = (imageWidth - drawnWidth) / 2.0;
	const double offsetY = (imageHeight - drawnHeight) / 2.0;


	// --------------------------------------------------------------
	// Conversion coordonnées VO -> coordonnées image.
	// 
	// L'axe vertical d'une image OpenCV descend.
	// On inverse donc Z pour avoir une représentation plus naturelle
	// de la trajectoire.
	// --------------------------------------------------------------
	auto worldToImage = 
		[&](double x, double z)
		{
			const int pixelX =
				static_cast<int>(
					offsetX + (x - minX) * scale
				);

			const int pixelY =
				static_cast<int>(
					imageHeight - (offsetY + (z - minZ) * scale)
				);

			return cv::Point(pixelX, pixelY);
		};

	// --------------------------------------------------------------
	// Dessin de la trajectoire
	// --------------------------------------------------------------
	for (std::size_t i = 1; i < trajectory.size(); ++i)
	{
		const cv::Point previous = 
			worldToImage(
				trajectory[i-1].x,
				trajectory[i-1].z
			);

		const cv::Point current =
			worldToImage(
				trajectory[i].x,
				trajectory[i].z
			);

		cv::line(
			visualization,
			previous,
			current,
			cv::Scalar(80, 80, 80),
			2,
			cv::LINE_AA
		);

		cv::circle(
			visualization,
			current,
			3,
			cv::Scalar(0, 0, 0),
			cv::FILLED,
			cv::LINE_AA
		);
	}

	// --------------------------------------------------------------
	// Départ et arrivée de la trajectoire
	// --------------------------------------------------------------
	const cv::Point startPoint = 
		worldToImage(
			trajectory.front().x,
			trajectory.front().z
		);

	const cv::Point endPoint =
		worldToImage(
			trajectory.back().x,
			trajectory.back().z
		);

	// Vert = départ
	cv::circle(
		visualization,
		startPoint,
		9,
		cv::Scalar(0, 180, 0),
		cv::FILLED,
		cv::LINE_AA
	);

	// Rouge = arrivée
	cv::circle(
		visualization,
		endPoint,
		9,
		cv::Scalar(0, 0, 200),
		cv::FILLED,
		cv::LINE_AA
	);


	cv::putText(
		visualization,
		"START",
		startPoint + cv::Point(12, -10),
		cv::FONT_HERSHEY_SIMPLEX,
		0.7,
		cv::Scalar(0, 120, 0),
		2,
		cv::LINE_AA
	);

	cv::putText(
		visualization,
		"END",
		endPoint + cv::Point(12, -10),
		cv::FONT_HERSHEY_SIMPLEX,
		0.7,
		cv::Scalar(0, 0, 180),
		2,
		cv::LINE_AA
	);

	cv::putText(
		visualization,
		trajectoryDisplayName + " - X/Z trajectory",
		cv::Point(25, 35),
		cv::FONT_HERSHEY_SIMPLEX,
		0.8,
		cv::Scalar(0, 0, 0),
		2,
		cv::LINE_AA
	);


	// --------------------------------------------------------------
	// Sauvegarde
	// --------------------------------------------------------------
	if (outputImagePath.has_parent_path())
	{
		fs::create_directories(
			outputImagePath.parent_path()
		);
	}

	if (!cv::imwrite(
		outputImagePath.string(),
		visualization
	))
	{
		std::cerr << "Error: could not save trajectory image.\n";
		return 1;
	}

	std::cout << "Trajectory visualization saved to: "
			  << outputImagePath.string()
			  << '\n';

	// Affichage à l'écran
	cv::imshow(
		trajectoryDisplayName,
		visualization
	);

	cv::waitKey(0);

	return 0;
}