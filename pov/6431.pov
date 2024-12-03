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
    sphere { m*<-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 1 }        
    sphere {  m*<0.15464060820173953,0.007115854642116531,8.999468252098602>, 1 }
    sphere {  m*<7.509992046201712,-0.08180442135224039,-5.580025037946747>, 1 }
    sphere {  m*<-4.840688559602512,3.8621029160256266,-2.68982703047441>, 1}
    sphere { m*<-2.603602710993565,-3.276285457272257,-1.517546306760873>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15464060820173953,0.007115854642116531,8.999468252098602>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5 }
    cylinder { m*<7.509992046201712,-0.08180442135224039,-5.580025037946747>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5}
    cylinder { m*<-4.840688559602512,3.8621029160256266,-2.68982703047441>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5 }
    cylinder {  m*<-2.603602710993565,-3.276285457272257,-1.517546306760873>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5}

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
    sphere { m*<-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 1 }        
    sphere {  m*<0.15464060820173953,0.007115854642116531,8.999468252098602>, 1 }
    sphere {  m*<7.509992046201712,-0.08180442135224039,-5.580025037946747>, 1 }
    sphere {  m*<-4.840688559602512,3.8621029160256266,-2.68982703047441>, 1}
    sphere { m*<-2.603602710993565,-3.276285457272257,-1.517546306760873>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15464060820173953,0.007115854642116531,8.999468252098602>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5 }
    cylinder { m*<7.509992046201712,-0.08180442135224039,-5.580025037946747>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5}
    cylinder { m*<-4.840688559602512,3.8621029160256266,-2.68982703047441>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5 }
    cylinder {  m*<-2.603602710993565,-3.276285457272257,-1.517546306760873>, <-1.2941079251386414,-0.6556043098033414,-0.8728926001107011>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    