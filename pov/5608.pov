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
    sphere { m*<-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 1 }        
    sphere {  m*<0.180116879271574,0.28328570088606364,8.590730308409809>, 1 }
    sphere {  m*<5.442136839827888,0.061756577635343146,-4.57888066981411>, 1 }
    sphere {  m*<-2.7043522033827325,2.1617494114549776,-2.2330073532549006>, 1}
    sphere { m*<-2.436564982344901,-2.7259425309489203,-2.04346106809233>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.180116879271574,0.28328570088606364,8.590730308409809>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5 }
    cylinder { m*<5.442136839827888,0.061756577635343146,-4.57888066981411>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5}
    cylinder { m*<-2.7043522033827325,2.1617494114549776,-2.2330073532549006>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5 }
    cylinder {  m*<-2.436564982344901,-2.7259425309489203,-2.04346106809233>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5}

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
    sphere { m*<-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 1 }        
    sphere {  m*<0.180116879271574,0.28328570088606364,8.590730308409809>, 1 }
    sphere {  m*<5.442136839827888,0.061756577635343146,-4.57888066981411>, 1 }
    sphere {  m*<-2.7043522033827325,2.1617494114549776,-2.2330073532549006>, 1}
    sphere { m*<-2.436564982344901,-2.7259425309489203,-2.04346106809233>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.180116879271574,0.28328570088606364,8.590730308409809>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5 }
    cylinder { m*<5.442136839827888,0.061756577635343146,-4.57888066981411>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5}
    cylinder { m*<-2.7043522033827325,2.1617494114549776,-2.2330073532549006>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5 }
    cylinder {  m*<-2.436564982344901,-2.7259425309489203,-2.04346106809233>, <-1.046450422076611,-0.167232136925715,-1.3235032521580965>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    