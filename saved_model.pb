▒Щ
п 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8 ▌
|
Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/v
u
&Adam/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output/bias/v*
_output_shapes
:*
dtype0
Е
Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*%
shared_nameAdam/Output/kernel/v
~
(Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/v*
_output_shapes
:	А*
dtype0
}
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/dense2/bias/v
v
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes	
:А*
dtype0
Е
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*%
shared_nameAdam/dense2/kernel/v
~
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes
:	@А*
dtype0
|
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/v
u
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АР@*%
shared_nameAdam/dense1/kernel/v

(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v* 
_output_shapes
:
АР@*
dtype0
{
Adam/conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv4/bias/v
t
%Adam/conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/v*
_output_shapes	
:А*
dtype0
М
Adam/conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameAdam/conv4/kernel/v
Е
'Adam/conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/v*(
_output_shapes
:АА*
dtype0
{
Adam/conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv3/bias/v
t
%Adam/conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/v*
_output_shapes	
:А*
dtype0
М
Adam/conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameAdam/conv3/kernel/v
Е
'Adam/conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/v*(
_output_shapes
:АА*
dtype0
{
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv2/bias/v
t
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes	
:А*
dtype0
Л
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameAdam/conv2/kernel/v
Д
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*'
_output_shapes
:@А*
dtype0
z
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv1/bias/v
s
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes
:@*
dtype0
К
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv1/kernel/v
Г
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/m
u
&Adam/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output/bias/m*
_output_shapes
:*
dtype0
Е
Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*%
shared_nameAdam/Output/kernel/m
~
(Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/m*
_output_shapes
:	А*
dtype0
}
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/dense2/bias/m
v
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes	
:А*
dtype0
Е
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*%
shared_nameAdam/dense2/kernel/m
~
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes
:	@А*
dtype0
|
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/m
u
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АР@*%
shared_nameAdam/dense1/kernel/m

(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m* 
_output_shapes
:
АР@*
dtype0
{
Adam/conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv4/bias/m
t
%Adam/conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/m*
_output_shapes	
:А*
dtype0
М
Adam/conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameAdam/conv4/kernel/m
Е
'Adam/conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/m*(
_output_shapes
:АА*
dtype0
{
Adam/conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv3/bias/m
t
%Adam/conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/m*
_output_shapes	
:А*
dtype0
М
Adam/conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameAdam/conv3/kernel/m
Е
'Adam/conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/m*(
_output_shapes
:АА*
dtype0
{
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/conv2/bias/m
t
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes	
:А*
dtype0
Л
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameAdam/conv2/kernel/m
Д
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*'
_output_shapes
:@А*
dtype0
z
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv1/bias/m
s
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes
:@*
dtype0
К
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv1/kernel/m
Г
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*&
_output_shapes
:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
w
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_nameOutput/kernel
p
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes
:	А*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:А*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	@А*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:@*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АР@*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
АР@*
dtype0
m

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
conv4/bias
f
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes	
:А*
dtype0
~
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*
shared_nameconv4/kernel
w
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*(
_output_shapes
:АА*
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:А*
dtype0
~
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*
shared_nameconv3/kernel
w
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*(
_output_shapes
:АА*
dtype0
m

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
conv2/bias
f
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes	
:А*
dtype0
}
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*
shared_nameconv2/kernel
v
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*'
_output_shapes
:@А*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:@*
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:@*
dtype0
О
serving_default_conv1_inputPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense1/kerneldense1/biasdense2/kerneldense2/biasOutput/kernelOutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_107438

NoOpNoOp
Єr
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*нr
valueгrBаr BЩr
а
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
О
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
╚
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
О
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
╚
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op*
О
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
╚
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op*
О
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
О
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
ж
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias*
ж
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias*
ж
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias*
j
0
1
+2
,3
:4
;5
I6
J7
^8
_9
f10
g11
n12
o13*
j
0
1
+2
,3
:4
;5
I6
J7
^8
_9
f10
g11
n12
o13*

p0
q1
r2* 
░
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
xtrace_0
ytrace_1
ztrace_2
{trace_3* 
6
|trace_0
}trace_1
~trace_2
trace_3* 
* 
с
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemюmя+mЁ,mё:mЄ;mєImЇJmї^mЎ_mўfm°gm∙nm·om√v№v¤+v■,v :vА;vБIvВJvГ^vД_vЕfvЖgvЗnvИovЙ*

Еserving_default* 

0
1*

0
1*
* 
Ш
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Лtrace_0* 

Мtrace_0* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

Тtrace_0* 

Уtrace_0* 

+0
,1*

+0
,1*
* 
Ш
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Щtrace_0* 

Ъtrace_0* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

аtrace_0* 

бtrace_0* 

:0
;1*

:0
;1*
* 
Ш
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

зtrace_0* 

иtrace_0* 
\V
VARIABLE_VALUEconv3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 

I0
J1*

I0
J1*
* 
Ш
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

╡trace_0* 

╢trace_0* 
\V
VARIABLE_VALUEconv4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╖non_trainable_variables
╕layers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 

╝trace_0* 

╜trace_0* 
* 
* 
* 
Ц
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

├trace_0* 

─trace_0* 

^0
_1*

^0
_1*
	
p0* 
Ш
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

╩trace_0* 

╦trace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

f0
g1*
	
q0* 
Ш
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

╤trace_0* 

╥trace_0* 
]W
VARIABLE_VALUEdense2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

n0
o1*
	
r0* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

╪trace_0* 

┘trace_0* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

┌trace_0* 

█trace_0* 

▄trace_0* 
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*

▌0
▐1
▀2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
p0* 
* 
* 
* 
* 
* 
* 
	
q0* 
* 
* 
* 
* 
* 
* 
	
r0* 
* 
* 
* 
* 
* 
* 
<
р	variables
с	keras_api

тtotal

уcount*
M
ф	variables
х	keras_api

цtotal

чcount
ш
_fn_kwargs*
M
щ	variables
ъ	keras_api

ыtotal

ьcount
э
_fn_kwargs*

т0
у1*

р	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ц0
ч1*

ф	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ы0
ь1*

щ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp'Adam/conv3/kernel/m/Read/ReadVariableOp%Adam/conv3/bias/m/Read/ReadVariableOp'Adam/conv4/kernel/m/Read/ReadVariableOp%Adam/conv4/bias/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp'Adam/conv3/kernel/v/Read/ReadVariableOp%Adam/conv3/bias/v/Read/ReadVariableOp'Adam/conv4/kernel/v/Read/ReadVariableOp%Adam/conv4/bias/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_108070
╜	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense1/kerneldense1/biasdense2/kerneldense2/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/conv1/kernel/mAdam/conv1/bias/mAdam/conv2/kernel/mAdam/conv2/bias/mAdam/conv3/kernel/mAdam/conv3/bias/mAdam/conv4/kernel/mAdam/conv4/bias/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/Output/kernel/mAdam/Output/bias/mAdam/conv1/kernel/vAdam/conv1/bias/vAdam/conv2/kernel/vAdam/conv2/bias/vAdam/conv3/kernel/vAdam/conv3/bias/vAdam/conv4/kernel/vAdam/conv4/bias/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense2/kernel/vAdam/dense2/bias/vAdam/Output/kernel/vAdam/Output/bias/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_108239∙ц	
З
ж
B__inference_Output_layer_call_and_return_conditional_losses_106992

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/Output/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         П
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┼
Ц
'__inference_dense2_layer_call_fn_107822

inputs
unknown:	@А
	unknown_0:	А
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_106971p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ф
Ф
3__inference_gun_CNN_classifier_layer_call_fn_107483

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:
АР@
	unknown_8:@
	unknown_9:	@А

unknown_10:	А

unknown_11:	А

unknown_12:
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
н
E
)__inference_maxpool2_layer_call_fn_107713

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┤
D
(__inference_flatten_layer_call_fn_107783

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106933b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         АР"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ж	
о
__inference_loss_fn_1_107879K
8dense2_kernel_regularizer_l2loss_readvariableop_resource:	@А
identityИв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpй
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8dense2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!dense2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp
·╨
ч
"__inference__traced_restore_108239
file_prefix7
assignvariableop_conv1_kernel:@+
assignvariableop_1_conv1_bias:@:
assignvariableop_2_conv2_kernel:@А,
assignvariableop_3_conv2_bias:	А;
assignvariableop_4_conv3_kernel:АА,
assignvariableop_5_conv3_bias:	А;
assignvariableop_6_conv4_kernel:АА,
assignvariableop_7_conv4_bias:	А4
 assignvariableop_8_dense1_kernel:
АР@,
assignvariableop_9_dense1_bias:@4
!assignvariableop_10_dense2_kernel:	@А.
assignvariableop_11_dense2_bias:	А4
!assignvariableop_12_output_kernel:	А-
assignvariableop_13_output_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: %
assignvariableop_19_total_2: %
assignvariableop_20_count_2: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: #
assignvariableop_23_total: #
assignvariableop_24_count: A
'assignvariableop_25_adam_conv1_kernel_m:@3
%assignvariableop_26_adam_conv1_bias_m:@B
'assignvariableop_27_adam_conv2_kernel_m:@А4
%assignvariableop_28_adam_conv2_bias_m:	АC
'assignvariableop_29_adam_conv3_kernel_m:АА4
%assignvariableop_30_adam_conv3_bias_m:	АC
'assignvariableop_31_adam_conv4_kernel_m:АА4
%assignvariableop_32_adam_conv4_bias_m:	А<
(assignvariableop_33_adam_dense1_kernel_m:
АР@4
&assignvariableop_34_adam_dense1_bias_m:@;
(assignvariableop_35_adam_dense2_kernel_m:	@А5
&assignvariableop_36_adam_dense2_bias_m:	А;
(assignvariableop_37_adam_output_kernel_m:	А4
&assignvariableop_38_adam_output_bias_m:A
'assignvariableop_39_adam_conv1_kernel_v:@3
%assignvariableop_40_adam_conv1_bias_v:@B
'assignvariableop_41_adam_conv2_kernel_v:@А4
%assignvariableop_42_adam_conv2_bias_v:	АC
'assignvariableop_43_adam_conv3_kernel_v:АА4
%assignvariableop_44_adam_conv3_bias_v:	АC
'assignvariableop_45_adam_conv4_kernel_v:АА4
%assignvariableop_46_adam_conv4_bias_v:	А<
(assignvariableop_47_adam_dense1_kernel_v:
АР@4
&assignvariableop_48_adam_dense1_bias_v:@;
(assignvariableop_49_adam_dense2_kernel_v:	@А5
&assignvariableop_50_adam_dense2_bias_v:	А;
(assignvariableop_51_adam_output_kernel_v:	А4
&assignvariableop_52_adam_output_bias_v:
identity_54ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╠
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Є
valueшBх6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▄
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B п
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapes█
╪::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_conv1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_conv1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_conv2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_conv2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_conv3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_conv3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_conv4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_conv4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_dense1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_dense2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_conv1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_conv1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_conv2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_conv2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_conv3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_conv3_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_conv4_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_conv4_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_dense1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_dense2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_output_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_output_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ▌	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_54IdentityIdentity_53:output:0^NoOp_1*
T0*
_output_shapes
: ╩	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
М
¤
A__inference_conv4_layer_call_and_return_conditional_losses_107768

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
М
`
D__inference_maxpool4_layer_call_and_return_conditional_losses_107778

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╟
Ц
'__inference_dense1_layer_call_fn_107798

inputs
unknown:
АР@
	unknown_0:@
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_106950o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АР: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         АР
 
_user_specified_nameinputs
┐C
┬
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107385
conv1_input&
conv1_107332:@
conv1_107334:@'
conv2_107338:@А
conv2_107340:	А(
conv3_107344:АА
conv3_107346:	А(
conv4_107350:АА
conv4_107352:	А!
dense1_107357:
АР@
dense1_107359:@ 
dense2_107362:	@А
dense2_107364:	А 
output_107367:	А
output_107369:
identityИвOutput/StatefulPartitionedCallв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвdense1/StatefulPartitionedCallв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/StatefulPartitionedCallв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpЇ
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_107332conv1_107334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_106866у
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809Л
conv2/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_107338conv2_107340*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         22А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_106884ф
maxpool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821Л
conv3/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_107344conv3_107346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_106902ф
maxpool3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833Л
conv4/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_107350conv4_107352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_106920ф
maxpool4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845╓
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106933Е
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_107357dense1_107359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_106950Н
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_107362dense2_107364*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_106971М
Output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_107367output_107369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_106992
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense1_107357* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense2_107362*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_107367*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp^Output/StatefulPartitionedCall0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
╚\
╕
!__inference__wrapped_model_106800
conv1_inputQ
7gun_cnn_classifier_conv1_conv2d_readvariableop_resource:@F
8gun_cnn_classifier_conv1_biasadd_readvariableop_resource:@R
7gun_cnn_classifier_conv2_conv2d_readvariableop_resource:@АG
8gun_cnn_classifier_conv2_biasadd_readvariableop_resource:	АS
7gun_cnn_classifier_conv3_conv2d_readvariableop_resource:ААG
8gun_cnn_classifier_conv3_biasadd_readvariableop_resource:	АS
7gun_cnn_classifier_conv4_conv2d_readvariableop_resource:ААG
8gun_cnn_classifier_conv4_biasadd_readvariableop_resource:	АL
8gun_cnn_classifier_dense1_matmul_readvariableop_resource:
АР@G
9gun_cnn_classifier_dense1_biasadd_readvariableop_resource:@K
8gun_cnn_classifier_dense2_matmul_readvariableop_resource:	@АH
9gun_cnn_classifier_dense2_biasadd_readvariableop_resource:	АK
8gun_cnn_classifier_output_matmul_readvariableop_resource:	АG
9gun_cnn_classifier_output_biasadd_readvariableop_resource:
identityИв0gun_CNN_classifier/Output/BiasAdd/ReadVariableOpв/gun_CNN_classifier/Output/MatMul/ReadVariableOpв/gun_CNN_classifier/conv1/BiasAdd/ReadVariableOpв.gun_CNN_classifier/conv1/Conv2D/ReadVariableOpв/gun_CNN_classifier/conv2/BiasAdd/ReadVariableOpв.gun_CNN_classifier/conv2/Conv2D/ReadVariableOpв/gun_CNN_classifier/conv3/BiasAdd/ReadVariableOpв.gun_CNN_classifier/conv3/Conv2D/ReadVariableOpв/gun_CNN_classifier/conv4/BiasAdd/ReadVariableOpв.gun_CNN_classifier/conv4/Conv2D/ReadVariableOpв0gun_CNN_classifier/dense1/BiasAdd/ReadVariableOpв/gun_CNN_classifier/dense1/MatMul/ReadVariableOpв0gun_CNN_classifier/dense2/BiasAdd/ReadVariableOpв/gun_CNN_classifier/dense2/MatMul/ReadVariableOpо
.gun_CNN_classifier/conv1/Conv2D/ReadVariableOpReadVariableOp7gun_cnn_classifier_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╨
gun_CNN_classifier/conv1/Conv2DConv2Dconv1_input6gun_CNN_classifier/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@*
paddingSAME*
strides
д
/gun_CNN_classifier/conv1/BiasAdd/ReadVariableOpReadVariableOp8gun_cnn_classifier_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 gun_CNN_classifier/conv1/BiasAddBiasAdd(gun_CNN_classifier/conv1/Conv2D:output:07gun_CNN_classifier/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@К
gun_CNN_classifier/conv1/ReluRelu)gun_CNN_classifier/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:         dd@╚
#gun_CNN_classifier/maxpool1/MaxPoolMaxPool+gun_CNN_classifier/conv1/Relu:activations:0*/
_output_shapes
:         22@*
ksize
*
paddingVALID*
strides
п
.gun_CNN_classifier/conv2/Conv2D/ReadVariableOpReadVariableOp7gun_cnn_classifier_conv2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Є
gun_CNN_classifier/conv2/Conv2DConv2D,gun_CNN_classifier/maxpool1/MaxPool:output:06gun_CNN_classifier/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22А*
paddingSAME*
strides
е
/gun_CNN_classifier/conv2/BiasAdd/ReadVariableOpReadVariableOp8gun_cnn_classifier_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╔
 gun_CNN_classifier/conv2/BiasAddBiasAdd(gun_CNN_classifier/conv2/Conv2D:output:07gun_CNN_classifier/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22АЛ
gun_CNN_classifier/conv2/ReluRelu)gun_CNN_classifier/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         22А╔
#gun_CNN_classifier/maxpool2/MaxPoolMaxPool+gun_CNN_classifier/conv2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
░
.gun_CNN_classifier/conv3/Conv2D/ReadVariableOpReadVariableOp7gun_cnn_classifier_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Є
gun_CNN_classifier/conv3/Conv2DConv2D,gun_CNN_classifier/maxpool2/MaxPool:output:06gun_CNN_classifier/conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
е
/gun_CNN_classifier/conv3/BiasAdd/ReadVariableOpReadVariableOp8gun_cnn_classifier_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╔
 gun_CNN_classifier/conv3/BiasAddBiasAdd(gun_CNN_classifier/conv3/Conv2D:output:07gun_CNN_classifier/conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЛ
gun_CNN_classifier/conv3/ReluRelu)gun_CNN_classifier/conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
#gun_CNN_classifier/maxpool3/MaxPoolMaxPool+gun_CNN_classifier/conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
░
.gun_CNN_classifier/conv4/Conv2D/ReadVariableOpReadVariableOp7gun_cnn_classifier_conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Є
gun_CNN_classifier/conv4/Conv2DConv2D,gun_CNN_classifier/maxpool3/MaxPool:output:06gun_CNN_classifier/conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
е
/gun_CNN_classifier/conv4/BiasAdd/ReadVariableOpReadVariableOp8gun_cnn_classifier_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╔
 gun_CNN_classifier/conv4/BiasAddBiasAdd(gun_CNN_classifier/conv4/Conv2D:output:07gun_CNN_classifier/conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЛ
gun_CNN_classifier/conv4/ReluRelu)gun_CNN_classifier/conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
#gun_CNN_classifier/maxpool4/MaxPoolMaxPool+gun_CNN_classifier/conv4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
q
 gun_CNN_classifier/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     H  ║
"gun_CNN_classifier/flatten/ReshapeReshape,gun_CNN_classifier/maxpool4/MaxPool:output:0)gun_CNN_classifier/flatten/Const:output:0*
T0*)
_output_shapes
:         АРк
/gun_CNN_classifier/dense1/MatMul/ReadVariableOpReadVariableOp8gun_cnn_classifier_dense1_matmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0┬
 gun_CNN_classifier/dense1/MatMulMatMul+gun_CNN_classifier/flatten/Reshape:output:07gun_CNN_classifier/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ж
0gun_CNN_classifier/dense1/BiasAdd/ReadVariableOpReadVariableOp9gun_cnn_classifier_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
!gun_CNN_classifier/dense1/BiasAddBiasAdd*gun_CNN_classifier/dense1/MatMul:product:08gun_CNN_classifier/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Д
gun_CNN_classifier/dense1/ReluRelu*gun_CNN_classifier/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:         @й
/gun_CNN_classifier/dense2/MatMul/ReadVariableOpReadVariableOp8gun_cnn_classifier_dense2_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0─
 gun_CNN_classifier/dense2/MatMulMatMul,gun_CNN_classifier/dense1/Relu:activations:07gun_CNN_classifier/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
0gun_CNN_classifier/dense2/BiasAdd/ReadVariableOpReadVariableOp9gun_cnn_classifier_dense2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0┼
!gun_CNN_classifier/dense2/BiasAddBiasAdd*gun_CNN_classifier/dense2/MatMul:product:08gun_CNN_classifier/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
gun_CNN_classifier/dense2/ReluRelu*gun_CNN_classifier/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:         Ай
/gun_CNN_classifier/Output/MatMul/ReadVariableOpReadVariableOp8gun_cnn_classifier_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0├
 gun_CNN_classifier/Output/MatMulMatMul,gun_CNN_classifier/dense2/Relu:activations:07gun_CNN_classifier/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ж
0gun_CNN_classifier/Output/BiasAdd/ReadVariableOpReadVariableOp9gun_cnn_classifier_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0─
!gun_CNN_classifier/Output/BiasAddBiasAdd*gun_CNN_classifier/Output/MatMul:product:08gun_CNN_classifier/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         К
!gun_CNN_classifier/Output/SoftmaxSoftmax*gun_CNN_classifier/Output/BiasAdd:output:0*
T0*'
_output_shapes
:         z
IdentityIdentity+gun_CNN_classifier/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp1^gun_CNN_classifier/Output/BiasAdd/ReadVariableOp0^gun_CNN_classifier/Output/MatMul/ReadVariableOp0^gun_CNN_classifier/conv1/BiasAdd/ReadVariableOp/^gun_CNN_classifier/conv1/Conv2D/ReadVariableOp0^gun_CNN_classifier/conv2/BiasAdd/ReadVariableOp/^gun_CNN_classifier/conv2/Conv2D/ReadVariableOp0^gun_CNN_classifier/conv3/BiasAdd/ReadVariableOp/^gun_CNN_classifier/conv3/Conv2D/ReadVariableOp0^gun_CNN_classifier/conv4/BiasAdd/ReadVariableOp/^gun_CNN_classifier/conv4/Conv2D/ReadVariableOp1^gun_CNN_classifier/dense1/BiasAdd/ReadVariableOp0^gun_CNN_classifier/dense1/MatMul/ReadVariableOp1^gun_CNN_classifier/dense2/BiasAdd/ReadVariableOp0^gun_CNN_classifier/dense2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2d
0gun_CNN_classifier/Output/BiasAdd/ReadVariableOp0gun_CNN_classifier/Output/BiasAdd/ReadVariableOp2b
/gun_CNN_classifier/Output/MatMul/ReadVariableOp/gun_CNN_classifier/Output/MatMul/ReadVariableOp2b
/gun_CNN_classifier/conv1/BiasAdd/ReadVariableOp/gun_CNN_classifier/conv1/BiasAdd/ReadVariableOp2`
.gun_CNN_classifier/conv1/Conv2D/ReadVariableOp.gun_CNN_classifier/conv1/Conv2D/ReadVariableOp2b
/gun_CNN_classifier/conv2/BiasAdd/ReadVariableOp/gun_CNN_classifier/conv2/BiasAdd/ReadVariableOp2`
.gun_CNN_classifier/conv2/Conv2D/ReadVariableOp.gun_CNN_classifier/conv2/Conv2D/ReadVariableOp2b
/gun_CNN_classifier/conv3/BiasAdd/ReadVariableOp/gun_CNN_classifier/conv3/BiasAdd/ReadVariableOp2`
.gun_CNN_classifier/conv3/Conv2D/ReadVariableOp.gun_CNN_classifier/conv3/Conv2D/ReadVariableOp2b
/gun_CNN_classifier/conv4/BiasAdd/ReadVariableOp/gun_CNN_classifier/conv4/BiasAdd/ReadVariableOp2`
.gun_CNN_classifier/conv4/Conv2D/ReadVariableOp.gun_CNN_classifier/conv4/Conv2D/ReadVariableOp2d
0gun_CNN_classifier/dense1/BiasAdd/ReadVariableOp0gun_CNN_classifier/dense1/BiasAdd/ReadVariableOp2b
/gun_CNN_classifier/dense1/MatMul/ReadVariableOp/gun_CNN_classifier/dense1/MatMul/ReadVariableOp2d
0gun_CNN_classifier/dense2/BiasAdd/ReadVariableOp0gun_CNN_classifier/dense2/BiasAdd/ReadVariableOp2b
/gun_CNN_classifier/dense2/MatMul/ReadVariableOp/gun_CNN_classifier/dense2/MatMul/ReadVariableOp:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
И
№
A__inference_conv2_layer_call_and_return_conditional_losses_106884

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         22Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         22Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         22@
 
_user_specified_nameinputs
М
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_107688

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
И	
п
__inference_loss_fn_0_107870L
8dense1_kernel_regularizer_l2loss_readvariableop_resource:
АР@
identityИв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpк
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8dense1_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!dense1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp
░C
╜
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107011

inputs&
conv1_106867:@
conv1_106869:@'
conv2_106885:@А
conv2_106887:	А(
conv3_106903:АА
conv3_106905:	А(
conv4_106921:АА
conv4_106923:	А!
dense1_106951:
АР@
dense1_106953:@ 
dense2_106972:	@А
dense2_106974:	А 
output_106993:	А
output_106995:
identityИвOutput/StatefulPartitionedCallв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвdense1/StatefulPartitionedCallв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/StatefulPartitionedCallв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpя
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_106867conv1_106869*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_106866у
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809Л
conv2/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_106885conv2_106887*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         22А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_106884ф
maxpool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821Л
conv3/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_106903conv3_106905*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_106902ф
maxpool3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833Л
conv4/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_106921conv4_106923*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_106920ф
maxpool4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845╓
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106933Е
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_106951dense1_106953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_106950Н
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_106972dense2_106974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_106971М
Output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_106993output_106995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_106992
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense1_106951* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense2_106972*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_106993*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp^Output/StatefulPartitionedCall0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
М
¤
A__inference_conv3_layer_call_and_return_conditional_losses_107738

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╔
_
C__inference_flatten_layer_call_and_return_conditional_losses_107789

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     H  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АРZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         АР"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
А
·
A__inference_conv1_layer_call_and_return_conditional_losses_106866

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         dd@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         dd@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
М
`
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ю
Ю
&__inference_conv3_layer_call_fn_107727

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_106902x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
н
E
)__inference_maxpool4_layer_call_fn_107773

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ч
Ы
&__inference_conv1_layer_call_fn_107667

inputs!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_106866w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         dd@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
г
Щ
3__inference_gun_CNN_classifier_layer_call_fn_107042
conv1_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:
АР@
	unknown_8:@
	unknown_9:	@А

unknown_10:	А

unknown_11:	А

unknown_12:
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
З
ж
B__inference_Output_layer_call_and_return_conditional_losses_107861

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/Output/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         П
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
г
Щ
3__inference_gun_CNN_classifier_layer_call_fn_107273
conv1_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:
АР@
	unknown_8:@
	unknown_9:	@А

unknown_10:	А

unknown_11:	А

unknown_12:
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107209o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
·g
╞
__inference__traced_save_108070
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop2
.savev2_adam_conv3_kernel_m_read_readvariableop0
,savev2_adam_conv3_bias_m_read_readvariableop2
.savev2_adam_conv4_kernel_m_read_readvariableop0
,savev2_adam_conv4_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop2
.savev2_adam_conv3_kernel_v_read_readvariableop0
,savev2_adam_conv3_bias_v_read_readvariableop2
.savev2_adam_conv4_kernel_v_read_readvariableop0
,savev2_adam_conv4_bias_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╔
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Є
valueшBх6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop.savev2_adam_conv3_kernel_m_read_readvariableop,savev2_adam_conv3_bias_m_read_readvariableop.savev2_adam_conv4_kernel_m_read_readvariableop,savev2_adam_conv4_bias_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop.savev2_adam_conv3_kernel_v_read_readvariableop,savev2_adam_conv3_bias_v_read_readvariableop.savev2_adam_conv4_kernel_v_read_readvariableop,savev2_adam_conv4_bias_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ж
_input_shapesЇ
ё: :@:@:@А:А:АА:А:АА:А:
АР@:@:	@А:А:	А:: : : : : : : : : : : :@:@:@А:А:АА:А:АА:А:
АР@:@:	@А:А:	А::@:@:@А:А:АА:А:АА:А:
АР@:@:	@А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:&	"
 
_output_shapes
:
АР@: 


_output_shapes
:@:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:. *
(
_output_shapes
:АА:!!

_output_shapes	
:А:&""
 
_output_shapes
:
АР@: #

_output_shapes
:@:%$!

_output_shapes
:	@А:!%

_output_shapes	
:А:%&!

_output_shapes
:	А: '

_output_shapes
::,((
&
_output_shapes
:@: )

_output_shapes
:@:-*)
'
_output_shapes
:@А:!+

_output_shapes	
:А:.,*
(
_output_shapes
:АА:!-

_output_shapes	
:А:..*
(
_output_shapes
:АА:!/

_output_shapes	
:А:&0"
 
_output_shapes
:
АР@: 1

_output_shapes
:@:%2!

_output_shapes
:	@А:!3

_output_shapes	
:А:%4!

_output_shapes
:	А: 5

_output_shapes
::6

_output_shapes
: 
ч
К
$__inference_signature_wrapper_107438
conv1_input!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:
АР@
	unknown_8:@
	unknown_9:	@А

unknown_10:	А

unknown_11:	А

unknown_12:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_106800o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
З
з
B__inference_dense1_layer_call_and_return_conditional_losses_107813

inputs2
matmul_readvariableop_resource:
АР@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Р
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         АР
 
_user_specified_nameinputs
М
`
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ы
Э
&__inference_conv2_layer_call_fn_107697

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         22А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_106884x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         22А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         22@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         22@
 
_user_specified_nameinputs
ю
Ю
&__inference_conv4_layer_call_fn_107757

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_106920x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
Ф
3__inference_gun_CNN_classifier_layer_call_fn_107516

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@А
	unknown_2:	А%
	unknown_3:АА
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:
АР@
	unknown_8:@
	unknown_9:	@А

unknown_10:	А

unknown_11:	А

unknown_12:
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107209o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
┐C
┬
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107329
conv1_input&
conv1_107276:@
conv1_107278:@'
conv2_107282:@А
conv2_107284:	А(
conv3_107288:АА
conv3_107290:	А(
conv4_107294:АА
conv4_107296:	А!
dense1_107301:
АР@
dense1_107303:@ 
dense2_107306:	@А
dense2_107308:	А 
output_107311:	А
output_107313:
identityИвOutput/StatefulPartitionedCallв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвdense1/StatefulPartitionedCallв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/StatefulPartitionedCallв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpЇ
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_107276conv1_107278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_106866у
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809Л
conv2/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_107282conv2_107284*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         22А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_106884ф
maxpool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821Л
conv3/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_107288conv3_107290*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_106902ф
maxpool3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833Л
conv4/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_107294conv4_107296*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_106920ф
maxpool4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845╓
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106933Е
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_107301dense1_107303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_106950Н
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_107306dense2_107308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_106971М
Output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_107311output_107313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_106992
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense1_107301* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense2_107306*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_107311*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp^Output/StatefulPartitionedCall0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:\ X
/
_output_shapes
:         dd
%
_user_specified_nameconv1_input
Ж	
о
__inference_loss_fn_2_107888K
8output_kernel_regularizer_l2loss_readvariableop_resource:	А
identityИв/Output/kernel/Regularizer/L2Loss/ReadVariableOpй
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp8output_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentity!Output/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp
М
`
D__inference_maxpool2_layer_call_and_return_conditional_losses_107718

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ж
з
B__inference_dense2_layer_call_and_return_conditional_losses_107837

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АП
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ай
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
░C
╜
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107209

inputs&
conv1_107156:@
conv1_107158:@'
conv2_107162:@А
conv2_107164:	А(
conv3_107168:АА
conv3_107170:	А(
conv4_107174:АА
conv4_107176:	А!
dense1_107181:
АР@
dense1_107183:@ 
dense2_107186:	@А
dense2_107188:	А 
output_107191:	А
output_107193:
identityИвOutput/StatefulPartitionedCallв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/StatefulPartitionedCallвconv2/StatefulPartitionedCallвconv3/StatefulPartitionedCallвconv4/StatefulPartitionedCallвdense1/StatefulPartitionedCallв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/StatefulPartitionedCallв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpя
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_107156conv1_107158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         dd@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_106866у
maxpool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809Л
conv2/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_107162conv2_107164*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         22А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_106884ф
maxpool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_106821Л
conv3/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_107168conv3_107170*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_106902ф
maxpool3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833Л
conv4/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_107174conv4_107176*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_106920ф
maxpool4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_106845╓
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         АР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106933Е
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_107181dense1_107183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_106950Н
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_107186dense2_107188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_106971М
Output/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0output_107191output_107193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_106992
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense1_107181* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense2_107186*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ~
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_107191*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp^Output/StatefulPartitionedCall0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^dense1/StatefulPartitionedCall0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/StatefulPartitionedCall0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
н
E
)__inference_maxpool1_layer_call_fn_107683

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
М
`
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╔
_
C__inference_flatten_layer_call_and_return_conditional_losses_106933

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     H  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АРZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         АР"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
М
¤
A__inference_conv3_layer_call_and_return_conditional_losses_106902

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
№
A__inference_conv2_layer_call_and_return_conditional_losses_107708

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         22Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         22Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         22@
 
_user_specified_nameinputs
З
з
B__inference_dense1_layer_call_and_return_conditional_losses_106950

inputs2
matmul_readvariableop_resource:
АР@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @Р
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         АР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         АР
 
_user_specified_nameinputs
М
¤
A__inference_conv4_layer_call_and_return_conditional_losses_106920

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
─
Х
'__inference_Output_layer_call_fn_107846

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_Output_layer_call_and_return_conditional_losses_106992o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
н
E
)__inference_maxpool3_layer_call_fn_107743

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_106833Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
А
·
A__inference_conv1_layer_call_and_return_conditional_losses_107678

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         dd@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         dd@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
М
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_106809

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦U
т
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107658

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@?
$conv2_conv2d_readvariableop_resource:@А4
%conv2_biasadd_readvariableop_resource:	А@
$conv3_conv2d_readvariableop_resource:АА4
%conv3_biasadd_readvariableop_resource:	А@
$conv4_conv2d_readvariableop_resource:АА4
%conv4_biasadd_readvariableop_resource:	А9
%dense1_matmul_readvariableop_resource:
АР@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@А5
&dense2_biasadd_readvariableop_resource:	А8
%output_matmul_readvariableop_resource:	А4
&output_biasadd_readvariableop_resource:
identityИвOutput/BiasAdd/ReadVariableOpвOutput/MatMul/ReadVariableOpв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/BiasAdd/ReadVariableOpвdense2/MatMul/ReadVariableOpв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpИ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0е
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@*
paddingSAME*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         dd@в
maxpool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:         22@*
ksize
*
paddingVALID*
strides
Й
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╣
conv2/Conv2DConv2Dmaxpool1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22А*
paddingSAME*
strides

conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22Аe

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:         22Аг
maxpool2/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
К
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╣
conv3/Conv2DConv2Dmaxpool2/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аe

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:         Аг
maxpool3/MaxPoolMaxPoolconv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
К
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╣
conv4/Conv2DConv2Dmaxpool3/MaxPool:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides

conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:         Аг
maxpool4/MaxPoolMaxPoolconv4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     H  Б
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         АРД
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Й
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @А
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:         @Г
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Л
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АБ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0М
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:         АГ
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0К
Output/MatMulMatMuldense2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:         Ч
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ц
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ц
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         Н
NoOpNoOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Ж
з
B__inference_dense2_layer_call_and_return_conditional_losses_106971

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АП
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ай
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╦U
т
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107587

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@?
$conv2_conv2d_readvariableop_resource:@А4
%conv2_biasadd_readvariableop_resource:	А@
$conv3_conv2d_readvariableop_resource:АА4
%conv3_biasadd_readvariableop_resource:	А@
$conv4_conv2d_readvariableop_resource:АА4
%conv4_biasadd_readvariableop_resource:	А9
%dense1_matmul_readvariableop_resource:
АР@4
&dense1_biasadd_readvariableop_resource:@8
%dense2_matmul_readvariableop_resource:	@А5
&dense2_biasadd_readvariableop_resource:	А8
%output_matmul_readvariableop_resource:	А4
&output_biasadd_readvariableop_resource:
identityИвOutput/BiasAdd/ReadVariableOpвOutput/MatMul/ReadVariableOpв/Output/kernel/Regularizer/L2Loss/ReadVariableOpвconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpв/dense1/kernel/Regularizer/L2Loss/ReadVariableOpвdense2/BiasAdd/ReadVariableOpвdense2/MatMul/ReadVariableOpв/dense2/kernel/Regularizer/L2Loss/ReadVariableOpИ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0е
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@*
paddingSAME*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         dd@d

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         dd@в
maxpool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:         22@*
ksize
*
paddingVALID*
strides
Й
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╣
conv2/Conv2DConv2Dmaxpool1/MaxPool:output:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22А*
paddingSAME*
strides

conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         22Аe

conv2/ReluReluconv2/BiasAdd:output:0*
T0*0
_output_shapes
:         22Аг
maxpool2/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
К
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╣
conv3/Conv2DConv2Dmaxpool2/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аe

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:         Аг
maxpool3/MaxPoolMaxPoolconv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
К
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╣
conv4/Conv2DConv2Dmaxpool3/MaxPool:output:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides

conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:         Аг
maxpool4/MaxPoolMaxPoolconv4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     H  Б
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:         АРД
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Й
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @А
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Л
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:         @Г
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Л
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АБ
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0М
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А_
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:         АГ
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0К
Output/MatMulMatMuldense2/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:         Ч
/dense1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
АР@*
dtype0Д
 dense1/kernel/Regularizer/L2LossL2Loss7dense1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense1/kernel/Regularizer/mulMul(dense1/kernel/Regularizer/mul/x:output:0)dense1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ц
/dense2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0Д
 dense2/kernel/Regularizer/L2LossL2Loss7dense2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
dense2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
dense2/kernel/Regularizer/mulMul(dense2/kernel/Regularizer/mul/x:output:0)dense2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ц
/Output/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Д
 Output/kernel/Regularizer/L2LossL2Loss7Output/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: d
Output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Ъ
Output/kernel/Regularizer/mulMul(Output/kernel/Regularizer/mul/x:output:0)Output/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         Н
NoOpNoOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp0^Output/kernel/Regularizer/L2Loss/ReadVariableOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp0^dense1/kernel/Regularizer/L2Loss/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp0^dense2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         dd: : : : : : : : : : : : : : 2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2b
/Output/kernel/Regularizer/L2Loss/ReadVariableOp/Output/kernel/Regularizer/L2Loss/ReadVariableOp2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2b
/dense1/kernel/Regularizer/L2Loss/ReadVariableOp/dense1/kernel/Regularizer/L2Loss/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2b
/dense2/kernel/Regularizer/L2Loss/ReadVariableOp/dense2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
М
`
D__inference_maxpool3_layer_call_and_return_conditional_losses_107748

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╣
serving_defaultе
K
conv1_input<
serving_default_conv1_input:0         dd:
Output0
StatefulPartitionedCall:0         tensorflow/serving/predict:Не
║
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
е
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
е
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op"
_tf_keras_layer
е
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op"
_tf_keras_layer
е
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
е
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias"
_tf_keras_layer
╗
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias"
_tf_keras_layer
╗
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias"
_tf_keras_layer
Ж
0
1
+2
,3
:4
;5
I6
J7
^8
_9
f10
g11
n12
o13"
trackable_list_wrapper
Ж
0
1
+2
,3
:4
;5
I6
J7
^8
_9
f10
g11
n12
o13"
trackable_list_wrapper
5
p0
q1
r2"
trackable_list_wrapper
╩
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Б
xtrace_0
ytrace_1
ztrace_2
{trace_32Ц
3__inference_gun_CNN_classifier_layer_call_fn_107042
3__inference_gun_CNN_classifier_layer_call_fn_107483
3__inference_gun_CNN_classifier_layer_call_fn_107516
3__inference_gun_CNN_classifier_layer_call_fn_107273┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zxtrace_0zytrace_1zztrace_2z{trace_3
э
|trace_0
}trace_1
~trace_2
trace_32В
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107587
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107658
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107329
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107385┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z|trace_0z}trace_1z~trace_2ztrace_3
╨B═
!__inference__wrapped_model_106800conv1_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemюmя+mЁ,mё:mЄ;mєImЇJmї^mЎ_mўfm°gm∙nm·om√v№v¤+v■,v :vА;vБIvВJvГ^vД_vЕfvЖgvЗnvИovЙ"
	optimizer
-
Еserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
Лtrace_02═
&__inference_conv1_layer_call_fn_107667в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0
З
Мtrace_02ш
A__inference_conv1_layer_call_and_return_conditional_losses_107678в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
&:$@2conv1/kernel
:@2
conv1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
я
Тtrace_02╨
)__inference_maxpool1_layer_call_fn_107683в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zТtrace_0
К
Уtrace_02ы
D__inference_maxpool1_layer_call_and_return_conditional_losses_107688в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ь
Щtrace_02═
&__inference_conv2_layer_call_fn_107697в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЩtrace_0
З
Ъtrace_02ш
A__inference_conv2_layer_call_and_return_conditional_losses_107708в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
':%@А2conv2/kernel
:А2
conv2/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
я
аtrace_02╨
)__inference_maxpool2_layer_call_fn_107713в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0
К
бtrace_02ы
D__inference_maxpool2_layer_call_and_return_conditional_losses_107718в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ь
зtrace_02═
&__inference_conv3_layer_call_fn_107727в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zзtrace_0
З
иtrace_02ш
A__inference_conv3_layer_call_and_return_conditional_losses_107738в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0
(:&АА2conv3/kernel
:А2
conv3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
я
оtrace_02╨
)__inference_maxpool3_layer_call_fn_107743в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zоtrace_0
К
пtrace_02ы
D__inference_maxpool3_layer_call_and_return_conditional_losses_107748в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ь
╡trace_02═
&__inference_conv4_layer_call_fn_107757в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╡trace_0
З
╢trace_02ш
A__inference_conv4_layer_call_and_return_conditional_losses_107768в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0
(:&АА2conv4/kernel
:А2
conv4/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╖non_trainable_variables
╕layers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
я
╝trace_02╨
)__inference_maxpool4_layer_call_fn_107773в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0
К
╜trace_02ы
D__inference_maxpool4_layer_call_and_return_conditional_losses_107778в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ю
├trace_02╧
(__inference_flatten_layer_call_fn_107783в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
Й
─trace_02ъ
C__inference_flatten_layer_call_and_return_conditional_losses_107789в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
▓
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
э
╩trace_02╬
'__inference_dense1_layer_call_fn_107798в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
И
╦trace_02щ
B__inference_dense1_layer_call_and_return_conditional_losses_107813в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
!:
АР@2dense1/kernel
:@2dense1/bias
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
▓
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
э
╤trace_02╬
'__inference_dense2_layer_call_fn_107822в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
И
╥trace_02щ
B__inference_dense2_layer_call_and_return_conditional_losses_107837в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
 :	@А2dense2/kernel
:А2dense2/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
э
╪trace_02╬
'__inference_Output_layer_call_fn_107846в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
И
┘trace_02щ
B__inference_Output_layer_call_and_return_conditional_losses_107861в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
 :	А2Output/kernel
:2Output/bias
╧
┌trace_02░
__inference_loss_fn_0_107870П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z┌trace_0
╧
█trace_02░
__inference_loss_fn_1_107879П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z█trace_0
╧
▄trace_02░
__inference_loss_fn_2_107888П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z▄trace_0
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
8
▌0
▐1
▀2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЙBЖ
3__inference_gun_CNN_classifier_layer_call_fn_107042conv1_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
3__inference_gun_CNN_classifier_layer_call_fn_107483inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
3__inference_gun_CNN_classifier_layer_call_fn_107516inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
3__inference_gun_CNN_classifier_layer_call_fn_107273conv1_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107587inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107658inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
дBб
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107329conv1_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
дBб
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107385conv1_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╧B╠
$__inference_signature_wrapper_107438conv1_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv1_layer_call_fn_107667inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv1_layer_call_and_return_conditional_losses_107678inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_maxpool1_layer_call_fn_107683inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_maxpool1_layer_call_and_return_conditional_losses_107688inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv2_layer_call_fn_107697inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv2_layer_call_and_return_conditional_losses_107708inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_maxpool2_layer_call_fn_107713inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_maxpool2_layer_call_and_return_conditional_losses_107718inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv3_layer_call_fn_107727inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv3_layer_call_and_return_conditional_losses_107738inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_maxpool3_layer_call_fn_107743inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_maxpool3_layer_call_and_return_conditional_losses_107748inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv4_layer_call_fn_107757inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv4_layer_call_and_return_conditional_losses_107768inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_maxpool4_layer_call_fn_107773inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_maxpool4_layer_call_and_return_conditional_losses_107778inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_flatten_layer_call_fn_107783inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_flatten_layer_call_and_return_conditional_losses_107789inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_dense1_layer_call_fn_107798inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense1_layer_call_and_return_conditional_losses_107813inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_dense2_layer_call_fn_107822inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense2_layer_call_and_return_conditional_losses_107837inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
r0"
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_Output_layer_call_fn_107846inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_Output_layer_call_and_return_conditional_losses_107861inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│B░
__inference_loss_fn_0_107870"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
│B░
__inference_loss_fn_1_107879"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
│B░
__inference_loss_fn_2_107888"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
R
р	variables
с	keras_api

тtotal

уcount"
_tf_keras_metric
c
ф	variables
х	keras_api

цtotal

чcount
ш
_fn_kwargs"
_tf_keras_metric
c
щ	variables
ъ	keras_api

ыtotal

ьcount
э
_fn_kwargs"
_tf_keras_metric
0
т0
у1"
trackable_list_wrapper
.
р	variables"
_generic_user_object
:  (2total
:  (2count
0
ц0
ч1"
trackable_list_wrapper
.
ф	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ы0
ь1"
trackable_list_wrapper
.
щ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
+:)@2Adam/conv1/kernel/m
:@2Adam/conv1/bias/m
,:*@А2Adam/conv2/kernel/m
:А2Adam/conv2/bias/m
-:+АА2Adam/conv3/kernel/m
:А2Adam/conv3/bias/m
-:+АА2Adam/conv4/kernel/m
:А2Adam/conv4/bias/m
&:$
АР@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
%:#	@А2Adam/dense2/kernel/m
:А2Adam/dense2/bias/m
%:#	А2Adam/Output/kernel/m
:2Adam/Output/bias/m
+:)@2Adam/conv1/kernel/v
:@2Adam/conv1/bias/v
,:*@А2Adam/conv2/kernel/v
:А2Adam/conv2/bias/v
-:+АА2Adam/conv3/kernel/v
:А2Adam/conv3/bias/v
-:+АА2Adam/conv4/kernel/v
:А2Adam/conv4/bias/v
&:$
АР@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
%:#	@А2Adam/dense2/kernel/v
:А2Adam/dense2/bias/v
%:#	А2Adam/Output/kernel/v
:2Adam/Output/bias/vг
B__inference_Output_layer_call_and_return_conditional_losses_107861]no0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ {
'__inference_Output_layer_call_fn_107846Pno0в-
&в#
!К
inputs         А
к "К         д
!__inference__wrapped_model_106800+,:;IJ^_fgno<в9
2в/
-К*
conv1_input         dd
к "/к,
*
Output К
Output         ▒
A__inference_conv1_layer_call_and_return_conditional_losses_107678l7в4
-в*
(К%
inputs         dd
к "-в*
#К 
0         dd@
Ъ Й
&__inference_conv1_layer_call_fn_107667_7в4
-в*
(К%
inputs         dd
к " К         dd@▓
A__inference_conv2_layer_call_and_return_conditional_losses_107708m+,7в4
-в*
(К%
inputs         22@
к ".в+
$К!
0         22А
Ъ К
&__inference_conv2_layer_call_fn_107697`+,7в4
-в*
(К%
inputs         22@
к "!К         22А│
A__inference_conv3_layer_call_and_return_conditional_losses_107738n:;8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Л
&__inference_conv3_layer_call_fn_107727a:;8в5
.в+
)К&
inputs         А
к "!К         А│
A__inference_conv4_layer_call_and_return_conditional_losses_107768nIJ8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Л
&__inference_conv4_layer_call_fn_107757aIJ8в5
.в+
)К&
inputs         А
к "!К         Ад
B__inference_dense1_layer_call_and_return_conditional_losses_107813^^_1в.
'в$
"К
inputs         АР
к "%в"
К
0         @
Ъ |
'__inference_dense1_layer_call_fn_107798Q^_1в.
'в$
"К
inputs         АР
к "К         @г
B__inference_dense2_layer_call_and_return_conditional_losses_107837]fg/в,
%в"
 К
inputs         @
к "&в#
К
0         А
Ъ {
'__inference_dense2_layer_call_fn_107822Pfg/в,
%в"
 К
inputs         @
к "К         Ак
C__inference_flatten_layer_call_and_return_conditional_losses_107789c8в5
.в+
)К&
inputs         А
к "'в$
К
0         АР
Ъ В
(__inference_flatten_layer_call_fn_107783V8в5
.в+
)К&
inputs         А
к "К         АР╧
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107329}+,:;IJ^_fgnoDвA
:в7
-К*
conv1_input         dd
p 

 
к "%в"
К
0         
Ъ ╧
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107385}+,:;IJ^_fgnoDвA
:в7
-К*
conv1_input         dd
p

 
к "%в"
К
0         
Ъ ╩
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107587x+,:;IJ^_fgno?в<
5в2
(К%
inputs         dd
p 

 
к "%в"
К
0         
Ъ ╩
N__inference_gun_CNN_classifier_layer_call_and_return_conditional_losses_107658x+,:;IJ^_fgno?в<
5в2
(К%
inputs         dd
p

 
к "%в"
К
0         
Ъ з
3__inference_gun_CNN_classifier_layer_call_fn_107042p+,:;IJ^_fgnoDвA
:в7
-К*
conv1_input         dd
p 

 
к "К         з
3__inference_gun_CNN_classifier_layer_call_fn_107273p+,:;IJ^_fgnoDвA
:в7
-К*
conv1_input         dd
p

 
к "К         в
3__inference_gun_CNN_classifier_layer_call_fn_107483k+,:;IJ^_fgno?в<
5в2
(К%
inputs         dd
p 

 
к "К         в
3__inference_gun_CNN_classifier_layer_call_fn_107516k+,:;IJ^_fgno?в<
5в2
(К%
inputs         dd
p

 
к "К         ;
__inference_loss_fn_0_107870^в

в 
к "К ;
__inference_loss_fn_1_107879fв

в 
к "К ;
__inference_loss_fn_2_107888nв

в 
к "К ч
D__inference_maxpool1_layer_call_and_return_conditional_losses_107688ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┐
)__inference_maxpool1_layer_call_fn_107683СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ч
D__inference_maxpool2_layer_call_and_return_conditional_losses_107718ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┐
)__inference_maxpool2_layer_call_fn_107713СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ч
D__inference_maxpool3_layer_call_and_return_conditional_losses_107748ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┐
)__inference_maxpool3_layer_call_fn_107743СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ч
D__inference_maxpool4_layer_call_and_return_conditional_losses_107778ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┐
)__inference_maxpool4_layer_call_fn_107773СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╖
$__inference_signature_wrapper_107438О+,:;IJ^_fgnoKвH
в 
Aк>
<
conv1_input-К*
conv1_input         dd"/к,
*
Output К
Output         