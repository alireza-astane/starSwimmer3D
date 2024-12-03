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
    sphere { m*<-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 1 }        
    sphere {  m*<-4.009670678072636e-18,-2.7100090415595738e-18,9.068452875827662>, 1 }
    sphere {  m*<9.428090415820634,-2.0660000809538413e-18,-3.123880457505677>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.123880457505677>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.123880457505677>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.009670678072636e-18,-2.7100090415595738e-18,9.068452875827662>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0660000809538413e-18,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5}

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
    sphere { m*<-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 1 }        
    sphere {  m*<-4.009670678072636e-18,-2.7100090415595738e-18,9.068452875827662>, 1 }
    sphere {  m*<9.428090415820634,-2.0660000809538413e-18,-3.123880457505677>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.123880457505677>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.123880457505677>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.009670678072636e-18,-2.7100090415595738e-18,9.068452875827662>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5 }
    cylinder { m*<9.428090415820634,-2.0660000809538413e-18,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.123880457505677>, <-2.0087000687613003e-18,3.1905958762347238e-18,0.20945287582765584>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    