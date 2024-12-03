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
    sphere { m*<-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 1 }        
    sphere {  m*<0.3669650410087384,-0.16872913562118988,9.10766511925447>, 1 }
    sphere {  m*<7.72231647900871,-0.2576494116155464,-5.471828170790868>, 1 }
    sphere {  m*<-5.854412825466809,4.881074320729805,-3.207499819986835>, 1}
    sphere { m*<-2.3277172315717896,-3.606808266118911,-1.3763362829149655>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3669650410087384,-0.16872913562118988,9.10766511925447>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5 }
    cylinder { m*<7.72231647900871,-0.2576494116155464,-5.471828170790868>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5}
    cylinder { m*<-5.854412825466809,4.881074320729805,-3.207499819986835>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5 }
    cylinder {  m*<-2.3277172315717896,-3.606808266118911,-1.3763362829149655>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5}

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
    sphere { m*<-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 1 }        
    sphere {  m*<0.3669650410087384,-0.16872913562118988,9.10766511925447>, 1 }
    sphere {  m*<7.72231647900871,-0.2576494116155464,-5.471828170790868>, 1 }
    sphere {  m*<-5.854412825466809,4.881074320729805,-3.207499819986835>, 1}
    sphere { m*<-2.3277172315717896,-3.606808266118911,-1.3763362829149655>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3669650410087384,-0.16872913562118988,9.10766511925447>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5 }
    cylinder { m*<7.72231647900871,-0.2576494116155464,-5.471828170790868>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5}
    cylinder { m*<-5.854412825466809,4.881074320729805,-3.207499819986835>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5 }
    cylinder {  m*<-2.3277172315717896,-3.606808266118911,-1.3763362829149655>, <-1.068969056697056,-0.9552233334922836,-0.7574741950139003>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    