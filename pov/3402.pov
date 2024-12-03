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
    sphere { m*<0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 1 }        
    sphere {  m*<0.45667774833558544,0.7447970886819835,2.984620118950556>, 1 }
    sphere {  m*<2.950651037600151,0.7181209858880324,-1.2321441776211786>, 1 }
    sphere {  m*<-1.405672716298997,2.9445609549202594,-0.9768804175859646>, 1}
    sphere { m*<-3.0237680926862582,-5.50812165867715,-1.880002608416365>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45667774833558544,0.7447970886819835,2.984620118950556>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5 }
    cylinder { m*<2.950651037600151,0.7181209858880324,-1.2321441776211786>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5}
    cylinder { m*<-1.405672716298997,2.9445609549202594,-0.9768804175859646>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5 }
    cylinder {  m*<-3.0237680926862582,-5.50812165867715,-1.880002608416365>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5}

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
    sphere { m*<0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 1 }        
    sphere {  m*<0.45667774833558544,0.7447970886819835,2.984620118950556>, 1 }
    sphere {  m*<2.950651037600151,0.7181209858880324,-1.2321441776211786>, 1 }
    sphere {  m*<-1.405672716298997,2.9445609549202594,-0.9768804175859646>, 1}
    sphere { m*<-3.0237680926862582,-5.50812165867715,-1.880002608416365>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.45667774833558544,0.7447970886819835,2.984620118950556>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5 }
    cylinder { m*<2.950651037600151,0.7181209858880324,-1.2321441776211786>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5}
    cylinder { m*<-1.405672716298997,2.9445609549202594,-0.9768804175859646>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5 }
    cylinder {  m*<-3.0237680926862582,-5.50812165867715,-1.880002608416365>, <0.21594264359389376,0.616087010501658,-0.0029346521699945283>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    