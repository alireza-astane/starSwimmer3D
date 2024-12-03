#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 1 }        
    sphere {  m*<0.21518451886010836,0.28828876127499803,2.8447004657473975>, 1 }
    sphere {  m*<2.7091578081246794,0.2616126584810473,-1.3720638308243402>, 1 }
    sphere {  m*<-1.647165945774475,2.488052627513275,-1.1168000707891257>, 1}
    sphere { m*<-2.102774226931518,-3.7671147984807583,-1.346384574575274>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21518451886010836,0.28828876127499803,2.8447004657473975>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5 }
    cylinder { m*<2.7091578081246794,0.2616126584810473,-1.3720638308243402>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5}
    cylinder { m*<-1.647165945774475,2.488052627513275,-1.1168000707891257>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5 }
    cylinder {  m*<-2.102774226931518,-3.7671147984807583,-1.346384574575274>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 1 }        
    sphere {  m*<0.21518451886010836,0.28828876127499803,2.8447004657473975>, 1 }
    sphere {  m*<2.7091578081246794,0.2616126584810473,-1.3720638308243402>, 1 }
    sphere {  m*<-1.647165945774475,2.488052627513275,-1.1168000707891257>, 1}
    sphere { m*<-2.102774226931518,-3.7671147984807583,-1.346384574575274>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21518451886010836,0.28828876127499803,2.8447004657473975>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5 }
    cylinder { m*<2.7091578081246794,0.2616126584810473,-1.3720638308243402>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5}
    cylinder { m*<-1.647165945774475,2.488052627513275,-1.1168000707891257>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5 }
    cylinder {  m*<-2.102774226931518,-3.7671147984807583,-1.346384574575274>, <-0.02555058588158332,0.1595786830946726,-0.1428543053731534>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    