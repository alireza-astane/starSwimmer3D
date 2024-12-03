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
    sphere { m*<-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 1 }        
    sphere {  m*<0.310499385600707,-0.121989759863994,9.07889134789824>, 1 }
    sphere {  m*<7.665850823600672,-0.2109100358583509,-5.500601942147096>, 1 }
    sphere {  m*<-5.593231291762899,4.623115243890156,-3.074151689381692>, 1}
    sphere { m*<-2.39970733931423,-3.523093406370519,-1.4131684329287617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.310499385600707,-0.121989759863994,9.07889134789824>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5 }
    cylinder { m*<7.665850823600672,-0.2109100358583509,-5.500601942147096>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5}
    cylinder { m*<-5.593231291762899,4.623115243890156,-3.074151689381692>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5 }
    cylinder {  m*<-2.39970733931423,-3.523093406370519,-1.4131684329287617>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5}

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
    sphere { m*<-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 1 }        
    sphere {  m*<0.310499385600707,-0.121989759863994,9.07889134789824>, 1 }
    sphere {  m*<7.665850823600672,-0.2109100358583509,-5.500601942147096>, 1 }
    sphere {  m*<-5.593231291762899,4.623115243890156,-3.074151689381692>, 1}
    sphere { m*<-2.39970733931423,-3.523093406370519,-1.4131684329287617>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.310499385600707,-0.121989759863994,9.07889134789824>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5 }
    cylinder { m*<7.665850823600672,-0.2109100358583509,-5.500601942147096>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5}
    cylinder { m*<-5.593231291762899,4.623115243890156,-3.074151689381692>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5 }
    cylinder {  m*<-2.39970733931423,-3.523093406370519,-1.4131684329287617>, <-1.1287273941410556,-0.8788120484603182,-0.7880905195045576>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    