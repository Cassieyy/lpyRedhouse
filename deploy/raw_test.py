from flask import Flask,abort,request,jsonify
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return '<h>RedHouse Project</h>'

@app.route('/add_tast/',methods=['POST'])    
def add_task():
    print('###############',123123123123)
    print(request.json)
    return jsonify({'result':'success'})
if __name__ == "__main__":
    app.run("0.0.0.0",5000,debug=False)