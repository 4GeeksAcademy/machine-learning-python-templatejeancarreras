# California Housing: K-Means Tutorial

See [README.es.md](README.es.md) for the complete project guide in Spanish and
[src/housing.ipynb](src/housing.ipynb) for the executed tutorial.

```bash
pip install -r requirements.txt
python src/app.py
python -m unittest discover -s tests -v
```

The application uses only `Latitude`, `Longitude`, and `MedInc`, splits data
80/20 before fitting a scaler, trains six K-Means clusters, and tunes an
interpretable decision tree using training-only cross-validation. Test accuracy
is 0.9944 and macro F1 is 0.9906 against K-Means pseudo-labels, not ground truth.

Data is loaded from a local `housing.csv` or downloaded using scikit-learn on the
first run. No database or environment variables are required. Saved models are
in `models/`, labeled datasets in `data/processed/`, and plots and interpretation
in [reports/analysis.md](reports/analysis.md). Earlier review datasets and
[src/explore.ipynb](src/explore.ipynb) are retained but are not used by this project.

## Original Template Reference

The following optional database setup belongs to the original template, not to
the K-Means tutorial.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.