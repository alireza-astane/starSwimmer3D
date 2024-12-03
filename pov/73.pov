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
    sphere { m*<-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 1 }        
    sphere {  m*<-4.627021079771812e-18,4.1641819081786975e-19,9.57755516033697>, 1 }
    sphere {  m*<9.428090415820634,6.805488503454599e-20,-3.237778172996357>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.237778172996357>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.237778172996357>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.627021079771812e-18,4.1641819081786975e-19,9.57755516033697>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5 }
    cylinder { m*<9.428090415820634,6.805488503454599e-20,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5}

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
    sphere { m*<-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 1 }        
    sphere {  m*<-4.627021079771812e-18,4.1641819081786975e-19,9.57755516033697>, 1 }
    sphere {  m*<9.428090415820634,6.805488503454599e-20,-3.237778172996357>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-3.237778172996357>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-3.237778172996357>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-4.627021079771812e-18,4.1641819081786975e-19,9.57755516033697>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5 }
    cylinder { m*<9.428090415820634,6.805488503454599e-20,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-3.237778172996357>, <-2.8695311541948566e-18,2.7449850718178618e-18,0.09555516033697754>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    