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
    sphere { m*<-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 1 }        
    sphere {  m*<0.22533644956921411,0.28422566013277606,8.551084333550609>, 1 }
    sphere {  m*<5.138047881321004,0.05230858743820643,-4.393126888906426>, 1 }
    sphere {  m*<-2.611675308612149,2.164814343447425,-2.284823539588134>, 1}
    sphere { m*<-2.3438880875743178,-2.7228775989564724,-2.0952772544255636>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22533644956921411,0.28422566013277606,8.551084333550609>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5 }
    cylinder { m*<5.138047881321004,0.05230858743820643,-4.393126888906426>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5}
    cylinder { m*<-2.611675308612149,2.164814343447425,-2.284823539588134>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5 }
    cylinder {  m*<-2.3438880875743178,-2.7228775989564724,-2.0952772544255636>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5}

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
    sphere { m*<-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 1 }        
    sphere {  m*<0.22533644956921411,0.28422566013277606,8.551084333550609>, 1 }
    sphere {  m*<5.138047881321004,0.05230858743820643,-4.393126888906426>, 1 }
    sphere {  m*<-2.611675308612149,2.164814343447425,-2.284823539588134>, 1}
    sphere { m*<-2.3438880875743178,-2.7228775989564724,-2.0952772544255636>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.22533644956921411,0.28422566013277606,8.551084333550609>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5 }
    cylinder { m*<5.138047881321004,0.05230858743820643,-4.393126888906426>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5}
    cylinder { m*<-2.611675308612149,2.164814343447425,-2.284823539588134>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5 }
    cylinder {  m*<-2.3438880875743178,-2.7228775989564724,-2.0952772544255636>, <-0.9574031921273682,-0.164104902310757,-1.3685753829878429>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    