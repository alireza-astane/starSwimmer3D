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
    sphere { m*<-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 1 }        
    sphere {  m*<0.04292545101627554,0.10026492931643133,8.942536299941871>, 1 }
    sphere {  m*<7.398276889016251,0.011344653322074405,-5.6369569901034815>, 1 }
    sphere {  m*<-4.261387444766026,3.252784802128325,-2.3938335253951735>, 1}
    sphere { m*<-2.755381111429488,-3.080854826778967,-1.5953160726008293>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.04292545101627554,0.10026492931643133,8.942536299941871>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5 }
    cylinder { m*<7.398276889016251,0.011344653322074405,-5.6369569901034815>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5}
    cylinder { m*<-4.261387444766026,3.252784802128325,-2.3938335253951735>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5 }
    cylinder {  m*<-2.755381111429488,-3.080854826778967,-1.5953160726008293>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5}

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
    sphere { m*<-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 1 }        
    sphere {  m*<0.04292545101627554,0.10026492931643133,8.942536299941871>, 1 }
    sphere {  m*<7.398276889016251,0.011344653322074405,-5.6369569901034815>, 1 }
    sphere {  m*<-4.261387444766026,3.252784802128325,-2.3938335253951735>, 1}
    sphere { m*<-2.755381111429488,-3.080854826778967,-1.5953160726008293>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.04292545101627554,0.10026492931643133,8.942536299941871>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5 }
    cylinder { m*<7.398276889016251,0.011344653322074405,-5.6369569901034815>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5}
    cylinder { m*<-4.261387444766026,3.252784802128325,-2.3938335253951735>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5 }
    cylinder {  m*<-2.755381111429488,-3.080854826778967,-1.5953160726008293>, <-1.4129305560723664,-0.48110875102334066,-0.9339075313864924>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    