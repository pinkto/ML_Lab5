# ML_Lab5
It is necessary to generate a dataset and save it in csv format, depending on the option. Build a model, which will contain an autocoder and a regression model.Schematically, it should look as follows: 
Train the model and break the trained model into 3: 
  Data encoding model (Input data -> Encoded data), 
  Data decoding model (Encoded data -> Decoded data),
  Regression model (Input data -> Regression result). 
As a result, present the source code, the generated data in csv format, the encoded and decoded data in csv format, the regression result in csv format (what should be and what the model outputs), and the 3 models themselves in h5 format.

X ∈ N(-5,10)
e ∈ N(0,0.3)

Attribute   1	          2	            3	          4	       5	     6	      7
Formula	-X^(3+e)	ln⁡〖(|X|)+e〗	sin⁡〖(3X)+e〗	exp(X)+e	X+4+e	-x+√((X)+e)	X+e
