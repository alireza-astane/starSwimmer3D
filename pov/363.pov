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
    sphere { m*<-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 1 }        
    sphere {  m*<-6.689988173964982e-18,-3.0270045395377447e-18,7.914047066136345>, 1 }
    sphere {  m*<9.428090415820634,-3.054769481991393e-18,-2.8712862671970028>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8712862671970028>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8712862671970028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.689988173964982e-18,-3.0270045395377447e-18,7.914047066136345>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5 }
    cylinder { m*<9.428090415820634,-3.054769481991393e-18,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5}

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
    sphere { m*<-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 1 }        
    sphere {  m*<-6.689988173964982e-18,-3.0270045395377447e-18,7.914047066136345>, 1 }
    sphere {  m*<9.428090415820634,-3.054769481991393e-18,-2.8712862671970028>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.8712862671970028>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.8712862671970028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-6.689988173964982e-18,-3.0270045395377447e-18,7.914047066136345>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5 }
    cylinder { m*<9.428090415820634,-3.054769481991393e-18,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.8712862671970028>, <-4.13191242938967e-18,-2.0666486160335017e-19,0.46204706613632907>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    