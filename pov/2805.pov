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
    sphere { m*<0.6522352705173808,0.9901213239988937,0.25151420544551817>, 1 }        
    sphere {  m*<0.8943481583579193,1.082781273179742,3.2402888431612205>, 1 }
    sphere {  m*<3.3875953474204543,1.0827812731797415,-0.9769933653293963>, 1 }
    sphere {  m*<-1.6745238626921766,4.408574785811367,-1.124205097114283>, 1}
    sphere { m*<-3.9340167981466556,-7.472024070290036,-2.4595085211984555>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8943481583579193,1.082781273179742,3.2402888431612205>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5 }
    cylinder { m*<3.3875953474204543,1.0827812731797415,-0.9769933653293963>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5}
    cylinder { m*<-1.6745238626921766,4.408574785811367,-1.124205097114283>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5 }
    cylinder {  m*<-3.9340167981466556,-7.472024070290036,-2.4595085211984555>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5}

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
    sphere { m*<0.6522352705173808,0.9901213239988937,0.25151420544551817>, 1 }        
    sphere {  m*<0.8943481583579193,1.082781273179742,3.2402888431612205>, 1 }
    sphere {  m*<3.3875953474204543,1.0827812731797415,-0.9769933653293963>, 1 }
    sphere {  m*<-1.6745238626921766,4.408574785811367,-1.124205097114283>, 1}
    sphere { m*<-3.9340167981466556,-7.472024070290036,-2.4595085211984555>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8943481583579193,1.082781273179742,3.2402888431612205>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5 }
    cylinder { m*<3.3875953474204543,1.0827812731797415,-0.9769933653293963>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5}
    cylinder { m*<-1.6745238626921766,4.408574785811367,-1.124205097114283>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5 }
    cylinder {  m*<-3.9340167981466556,-7.472024070290036,-2.4595085211984555>, <0.6522352705173808,0.9901213239988937,0.25151420544551817>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    