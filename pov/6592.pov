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
    sphere { m*<-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 1 }        
    sphere {  m*<0.2772500198150032,-0.09447232110856502,9.061948163119983>, 1 }
    sphere {  m*<7.63260145781497,-0.18339259710292213,-5.517545126925361>, 1 }
    sphere {  m*<-5.4368967663991405,4.467336855183239,-2.994325572423598>, 1}
    sphere { m*<-2.4425341718831586,-3.472512933957422,-1.4350845947238924>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2772500198150032,-0.09447232110856502,9.061948163119983>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5 }
    cylinder { m*<7.63260145781497,-0.18339259710292213,-5.517545126925361>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5}
    cylinder { m*<-5.4368967663991405,4.467336855183239,-2.994325572423598>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5 }
    cylinder {  m*<-2.4425341718831586,-3.472512933957422,-1.4350845947238924>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5}

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
    sphere { m*<-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 1 }        
    sphere {  m*<0.2772500198150032,-0.09447232110856502,9.061948163119983>, 1 }
    sphere {  m*<7.63260145781497,-0.18339259710292213,-5.517545126925361>, 1 }
    sphere {  m*<-5.4368967663991405,4.467336855183239,-2.994325572423598>, 1}
    sphere { m*<-2.4425341718831586,-3.472512933957422,-1.4350845947238924>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2772500198150032,-0.09447232110856502,9.061948163119983>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5 }
    cylinder { m*<7.63260145781497,-0.18339259710292213,-5.517545126925361>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5}
    cylinder { m*<-5.4368967663991405,4.467336855183239,-2.994325572423598>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5 }
    cylinder {  m*<-2.4425341718831586,-3.472512933957422,-1.4350845947238924>, <-1.1639570926220575,-0.8328139633796495,-0.806145759431538>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    