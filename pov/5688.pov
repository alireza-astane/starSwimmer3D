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
    sphere { m*<-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 1 }        
    sphere {  m*<0.12220650435278008,0.28207742834139654,8.641386120382421>, 1 }
    sphere {  m*<5.813477550270601,0.07315159565443569,-4.809399261537898>, 1 }
    sphere {  m*<-2.8190513105229966,2.158009726095999,-2.167493886629366>, 1}
    sphere { m*<-2.5512640894851653,-2.729682216307898,-1.9779476014667952>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12220650435278008,0.28207742834139654,8.641386120382421>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5 }
    cylinder { m*<5.813477550270601,0.07315159565443569,-4.809399261537898>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5}
    cylinder { m*<-2.8190513105229966,2.158009726095999,-2.167493886629366>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5 }
    cylinder {  m*<-2.5512640894851653,-2.729682216307898,-1.9779476014667952>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5}

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
    sphere { m*<-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 1 }        
    sphere {  m*<0.12220650435278008,0.28207742834139654,8.641386120382421>, 1 }
    sphere {  m*<5.813477550270601,0.07315159565443569,-4.809399261537898>, 1 }
    sphere {  m*<-2.8190513105229966,2.158009726095999,-2.167493886629366>, 1}
    sphere { m*<-2.5512640894851653,-2.729682216307898,-1.9779476014667952>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12220650435278008,0.28207742834139654,8.641386120382421>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5 }
    cylinder { m*<5.813477550270601,0.07315159565443569,-4.809399261537898>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5}
    cylinder { m*<-2.8190513105229966,2.158009726095999,-2.167493886629366>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5 }
    cylinder {  m*<-2.5512640894851653,-2.729682216307898,-1.9779476014667952>, <-1.1568400006311346,-0.17104942617926677,-1.2660909525364108>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    