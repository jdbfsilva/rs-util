import rs
from sklearn.ensemble import RandomForestClassifier

# Input data
raster_file = "the/image/that/will/be/classfied/here"
training_dir = "your/data/training/here"
output_file = "the/name/of/the/map/here"

# Read the problem as supervised learning problem
df = rs.prepare_problem(raster_file, training_dir)

# Classifying one image
classifier = RandomForestClassifier(n_jobs=4, n_estimators=10)
classifier.fit(df["training_samples"], df["training_labels"])

result = classifier.predict(df["flat_pixels"])

# Save the image
rs.write_geotiff(output_file, result, 
                 df["cols"], df["rows"], df["geo_transform"], df["projection"])
