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
    sphere { m*<0.9119816917842044,0.6111052536769023,0.4050917304972911>, 1 }        
    sphere {  m*<1.1555716313956705,0.6625491129890126,3.394741344137657>, 1 }
    sphere {  m*<3.648818820458206,0.6625491129890124,-0.8225408643529604>, 1 }
    sphere {  m*<-2.5542153265590084,5.961030861391105,-1.6443423754719555>, 1}
    sphere { m*<-3.846433810708995,-7.721736301916813,-2.407719086803569>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1555716313956705,0.6625491129890126,3.394741344137657>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5 }
    cylinder { m*<3.648818820458206,0.6625491129890124,-0.8225408643529604>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5}
    cylinder { m*<-2.5542153265590084,5.961030861391105,-1.6443423754719555>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5 }
    cylinder {  m*<-3.846433810708995,-7.721736301916813,-2.407719086803569>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5}

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
    sphere { m*<0.9119816917842044,0.6111052536769023,0.4050917304972911>, 1 }        
    sphere {  m*<1.1555716313956705,0.6625491129890126,3.394741344137657>, 1 }
    sphere {  m*<3.648818820458206,0.6625491129890124,-0.8225408643529604>, 1 }
    sphere {  m*<-2.5542153265590084,5.961030861391105,-1.6443423754719555>, 1}
    sphere { m*<-3.846433810708995,-7.721736301916813,-2.407719086803569>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1555716313956705,0.6625491129890126,3.394741344137657>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5 }
    cylinder { m*<3.648818820458206,0.6625491129890124,-0.8225408643529604>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5}
    cylinder { m*<-2.5542153265590084,5.961030861391105,-1.6443423754719555>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5 }
    cylinder {  m*<-3.846433810708995,-7.721736301916813,-2.407719086803569>, <0.9119816917842044,0.6111052536769023,0.4050917304972911>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    