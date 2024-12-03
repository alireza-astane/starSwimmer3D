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
    sphere { m*<-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 1 }        
    sphere {  m*<0.3588210718638667,-0.161986537732929,9.103515106286956>, 1 }
    sphere {  m*<7.7141725098638405,-0.2509068137272854,-5.4759781837583805>, 1 }
    sphere {  m*<-5.817056420743752,4.844344297239626,-3.1884282472334338>, 1}
    sphere { m*<-2.3380452132417893,-3.5948939023184203,-1.381619782031203>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3588210718638667,-0.161986537732929,9.103515106286956>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5 }
    cylinder { m*<7.7141725098638405,-0.2509068137272854,-5.4759781837583805>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5}
    cylinder { m*<-5.817056420743752,4.844344297239626,-3.1884282472334338>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5 }
    cylinder {  m*<-2.3380452132417893,-3.5948939023184203,-1.381619782031203>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5}

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
    sphere { m*<-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 1 }        
    sphere {  m*<0.3588210718638667,-0.161986537732929,9.103515106286956>, 1 }
    sphere {  m*<7.7141725098638405,-0.2509068137272854,-5.4759781837583805>, 1 }
    sphere {  m*<-5.817056420743752,4.844344297239626,-3.1884282472334338>, 1}
    sphere { m*<-2.3380452132417893,-3.5948939023184203,-1.381619782031203>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3588210718638667,-0.161986537732929,9.103515106286956>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5 }
    cylinder { m*<7.7141725098638405,-0.2509068137272854,-5.4759781837583805>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5}
    cylinder { m*<-5.817056420743752,4.844344297239626,-3.1884282472334338>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5 }
    cylinder {  m*<-2.3380452132417893,-3.5948939023184203,-1.381619782031203>, <-1.0775823291633353,-0.9443273186741235,-0.7618863630201641>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    