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
    sphere { m*<-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 1 }        
    sphere {  m*<0.12404384692824921,0.03254122010518046,8.98387614845634>, 1 }
    sphere {  m*<7.479395284928217,-0.056379055889176743,-5.595617141589008>, 1 }
    sphere {  m*<-4.6860774056641175,3.701797713112278,-2.6108427286924885>, 1}
    sphere { m*<-2.6446530640804418,-3.224549600723802,-1.5385732565726924>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12404384692824921,0.03254122010518046,8.98387614845634>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5 }
    cylinder { m*<7.479395284928217,-0.056379055889176743,-5.595617141589008>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5}
    cylinder { m*<-4.6860774056641175,3.701797713112278,-2.6108427286924885>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5 }
    cylinder {  m*<-2.6446530640804418,-3.224549600723802,-1.5385732565726924>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5}

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
    sphere { m*<-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 1 }        
    sphere {  m*<0.12404384692824921,0.03254122010518046,8.98387614845634>, 1 }
    sphere {  m*<7.479395284928217,-0.056379055889176743,-5.595617141589008>, 1 }
    sphere {  m*<-4.6860774056641175,3.701797713112278,-2.6108427286924885>, 1}
    sphere { m*<-2.6446530640804418,-3.224549600723802,-1.5385732565726924>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12404384692824921,0.03254122010518046,8.98387614845634>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5 }
    cylinder { m*<7.479395284928217,-0.056379055889176743,-5.595617141589008>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5}
    cylinder { m*<-4.6860774056641175,3.701797713112278,-2.6108427286924885>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5 }
    cylinder {  m*<-2.6446530640804418,-3.224549600723802,-1.5385732565726924>, <-1.326635703515602,-0.609216128127134,-0.8895870465297108>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    