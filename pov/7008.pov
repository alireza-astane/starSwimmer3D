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
    sphere { m*<-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 1 }        
    sphere {  m*<0.6179675440680824,-0.2754583692711704,9.228700369694161>, 1 }
    sphere {  m*<7.985754742390892,-0.5605506200634324,-5.341977059379784>, 1 }
    sphere {  m*<-6.910208451298114,5.962530753557226,-3.851170156198177>, 1}
    sphere { m*<-2.056145872018222,-3.9984244630200134,-1.2017395589004338>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6179675440680824,-0.2754583692711704,9.228700369694161>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5 }
    cylinder { m*<7.985754742390892,-0.5605506200634324,-5.341977059379784>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5}
    cylinder { m*<-6.910208451298114,5.962530753557226,-3.851170156198177>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5 }
    cylinder {  m*<-2.056145872018222,-3.9984244630200134,-1.2017395589004338>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5}

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
    sphere { m*<-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 1 }        
    sphere {  m*<0.6179675440680824,-0.2754583692711704,9.228700369694161>, 1 }
    sphere {  m*<7.985754742390892,-0.5605506200634324,-5.341977059379784>, 1 }
    sphere {  m*<-6.910208451298114,5.962530753557226,-3.851170156198177>, 1}
    sphere { m*<-2.056145872018222,-3.9984244630200134,-1.2017395589004338>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6179675440680824,-0.2754583692711704,9.228700369694161>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5 }
    cylinder { m*<7.985754742390892,-0.5605506200634324,-5.341977059379784>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5}
    cylinder { m*<-6.910208451298114,5.962530753557226,-3.851170156198177>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5 }
    cylinder {  m*<-2.056145872018222,-3.9984244630200134,-1.2017395589004338>, <-0.801199950132081,-1.2653972831510887,-0.6205897273409984>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    