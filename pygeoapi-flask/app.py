import logging, psycopg2, datetime, io, os, subprocess, json
import pandas as pd

from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from flask import Flask, Response, request, jsonify
from pygeoapi.flask_app import BLUEPRINT as pygeoapi_blueprint

app = Flask(__name__)
CORS(app)

app.register_blueprint(pygeoapi_blueprint, url_prefix='/oapi')

api = Api(app)

def get_user_id(token):
	DB_NAME=os.getenv('SURFACE_DB_NAME')
	DB_USER=os.getenv('SURFACE_DB_USER')
	DB_PASSWORD=os.getenv('SURFACE_DB_PASSWORD')
	DB_HOST=os.getenv('SURFACE_DB_HOST')

	config = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}"

	with psycopg2.connect(config) as conn:
		with conn.cursor() as cursor:
			query = f"""
			SELECT user_id
			FROM authtoken_token
			WHERE key = '{token}'
			"""
			logging.info(query)        
			cursor.execute(query)
			data = cursor.fetchall()
			
	logging.info(data)
	logging.info(len(data))

	if len(data) > 0:
		if len(data[0]) > 0:
			user_id = data[0][0]
			return user_id
		return None
	return None

class DataExport(Resource):
	def post(self):
		logging.basicConfig(level=logging.INFO)

		# User Authentication
		auth_header = request.headers.get('Authorization')
		content_type_header = request.headers.get('Content-Type')

		token = auth_header.replace('Token ', '')

		user_id = get_user_id(token)
		is_authenticated = user_id is None
		if is_authenticated:
			response = jsonify({'message': 'Authentication failed'})
			response.status_code = 401  # Unauthorized
			return response		

		data_source = request.args.get('data_source')
		file_format = request.args.get('file_format')
		interval = request.args.get('interval')

		initial_date = request.args.get('initial_date')
		initial_time = request.args.get('initial_time')
		final_date = request.args.get('final_date')
		final_time = request.args.get('final_time')

		series = request.get_json()
		series_str = json.dumps(series)  # Convert series to string		

		# Call the standalone script with subprocess
		output = subprocess.check_output(['python', 'scripts/data_export.py', 
			'--data_source', data_source, 
			'--file_format', file_format, 
			'--interval', interval, 
			'--initial_date', initial_date, 
			'--initial_time', initial_time, 
			'--final_date', final_date, 
			'--final_time', final_time, 
			'--series', series_str  # Pass series as string			
		])

		output_str = output.decode('utf-8')
		df = pd.read_csv(io.StringIO(output_str))

		# print('Flask dataframe',df)

		try:
			# logging.info(df)

			# Download File
			if file_format == 'csv':
				output = io.StringIO()
				df.to_csv(output, index=False)
				output.seek(0)
			elif file_format == 'excell':
				output = io.BytesIO()
				df.to_excel(output, index=False)
				output.seek(0)
			elif file_format == 'rinstat':
				output = io.StringIO()
				df.to_csv(output, sep='\t', index=False)
				output.seek(0)

			return Response(
			  output,
			  mimetype="text/csv",
			  headers={"Content-Disposition": "attachment;filename=output.csv"}
			)

		except (psycopg2.DatabaseError, Exception) as error:
			return str(error)		

api.add_resource(DataExport, '/data_export')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)            