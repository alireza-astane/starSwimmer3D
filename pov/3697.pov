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
    sphere { m*<0.019110154256771983,0.244003409309984,-0.11697815437079334>, 1 }        
    sphere {  m*<0.25984525899846367,0.3727134874903094,2.870576616749758>, 1 }
    sphere {  m*<2.753818548263033,0.3460373846963586,-1.3461876798219792>, 1 }
    sphere {  m*<-1.602505205636119,2.572477353728587,-1.0909239197867646>, 1}
    sphere { m*<-2.288715546444809,-4.118610204588938,-1.4541178017094092>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25984525899846367,0.3727134874903094,2.870576616749758>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5 }
    cylinder { m*<2.753818548263033,0.3460373846963586,-1.3461876798219792>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5}
    cylinder { m*<-1.602505205636119,2.572477353728587,-1.0909239197867646>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5 }
    cylinder {  m*<-2.288715546444809,-4.118610204588938,-1.4541178017094092>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5}

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
    sphere { m*<0.019110154256771983,0.244003409309984,-0.11697815437079334>, 1 }        
    sphere {  m*<0.25984525899846367,0.3727134874903094,2.870576616749758>, 1 }
    sphere {  m*<2.753818548263033,0.3460373846963586,-1.3461876798219792>, 1 }
    sphere {  m*<-1.602505205636119,2.572477353728587,-1.0909239197867646>, 1}
    sphere { m*<-2.288715546444809,-4.118610204588938,-1.4541178017094092>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25984525899846367,0.3727134874903094,2.870576616749758>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5 }
    cylinder { m*<2.753818548263033,0.3460373846963586,-1.3461876798219792>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5}
    cylinder { m*<-1.602505205636119,2.572477353728587,-1.0909239197867646>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5 }
    cylinder {  m*<-2.288715546444809,-4.118610204588938,-1.4541178017094092>, <0.019110154256771983,0.244003409309984,-0.11697815437079334>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    