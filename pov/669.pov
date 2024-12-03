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
    sphere { m*<1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 1 }        
    sphere {  m*<-1.159884455658499e-19,-5.792095525688081e-18,6.141039132551844>, 1 }
    sphere {  m*<9.428090415820634,1.0785452732597138e-19,-2.5022942007815105>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.5022942007815105>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.5022942007815105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.159884455658499e-19,-5.792095525688081e-18,6.141039132551844>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5 }
    cylinder { m*<9.428090415820634,1.0785452732597138e-19,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5}

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
    sphere { m*<1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 1 }        
    sphere {  m*<-1.159884455658499e-19,-5.792095525688081e-18,6.141039132551844>, 1 }
    sphere {  m*<9.428090415820634,1.0785452732597138e-19,-2.5022942007815105>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.5022942007815105>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.5022942007815105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-1.159884455658499e-19,-5.792095525688081e-18,6.141039132551844>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5 }
    cylinder { m*<9.428090415820634,1.0785452732597138e-19,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.5022942007815105>, <1.7942149896356235e-18,-5.4695657506898965e-18,0.8310391325518234>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    