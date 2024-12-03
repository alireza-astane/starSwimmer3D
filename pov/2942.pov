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
    sphere { m*<0.5514868737931152,1.1246363704490063,0.19194629368224253>, 1 }        
    sphere {  m*<0.7927664745753502,1.2338786877860837,3.180227591435048>, 1 }
    sphere {  m*<3.286013663637885,1.2338786877860832,-1.0370546170555675>, 1 }
    sphere {  m*<-1.2852277636774634,3.7704734617774243,-0.8940271332302447>, 1}
    sphere { m*<-3.9647023777362125,-7.388222448153028,-2.4776533136115955>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7927664745753502,1.2338786877860837,3.180227591435048>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5 }
    cylinder { m*<3.286013663637885,1.2338786877860832,-1.0370546170555675>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5}
    cylinder { m*<-1.2852277636774634,3.7704734617774243,-0.8940271332302447>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5 }
    cylinder {  m*<-3.9647023777362125,-7.388222448153028,-2.4776533136115955>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5}

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
    sphere { m*<0.5514868737931152,1.1246363704490063,0.19194629368224253>, 1 }        
    sphere {  m*<0.7927664745753502,1.2338786877860837,3.180227591435048>, 1 }
    sphere {  m*<3.286013663637885,1.2338786877860832,-1.0370546170555675>, 1 }
    sphere {  m*<-1.2852277636774634,3.7704734617774243,-0.8940271332302447>, 1}
    sphere { m*<-3.9647023777362125,-7.388222448153028,-2.4776533136115955>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7927664745753502,1.2338786877860837,3.180227591435048>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5 }
    cylinder { m*<3.286013663637885,1.2338786877860832,-1.0370546170555675>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5}
    cylinder { m*<-1.2852277636774634,3.7704734617774243,-0.8940271332302447>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5 }
    cylinder {  m*<-3.9647023777362125,-7.388222448153028,-2.4776533136115955>, <0.5514868737931152,1.1246363704490063,0.19194629368224253>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    