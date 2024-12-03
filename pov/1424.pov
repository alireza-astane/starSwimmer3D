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
    sphere { m*<0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 1 }        
    sphere {  m*<0.6699126993096308,-1.2548276867627031e-18,3.972530280795805>, 1 }
    sphere {  m*<7.134799158653544,2.079812408927779e-18,-1.5598320094351446>, 1 }
    sphere {  m*<-4.226493872134569,8.164965809277259,-2.2209066638373622>, 1}
    sphere { m*<-4.226493872134569,-8.164965809277259,-2.220906663837366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6699126993096308,-1.2548276867627031e-18,3.972530280795805>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5 }
    cylinder { m*<7.134799158653544,2.079812408927779e-18,-1.5598320094351446>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5}
    cylinder { m*<-4.226493872134569,8.164965809277259,-2.2209066638373622>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5 }
    cylinder {  m*<-4.226493872134569,-8.164965809277259,-2.220906663837366>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5}

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
    sphere { m*<0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 1 }        
    sphere {  m*<0.6699126993096308,-1.2548276867627031e-18,3.972530280795805>, 1 }
    sphere {  m*<7.134799158653544,2.079812408927779e-18,-1.5598320094351446>, 1 }
    sphere {  m*<-4.226493872134569,8.164965809277259,-2.2209066638373622>, 1}
    sphere { m*<-4.226493872134569,-8.164965809277259,-2.220906663837366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6699126993096308,-1.2548276867627031e-18,3.972530280795805>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5 }
    cylinder { m*<7.134799158653544,2.079812408927779e-18,-1.5598320094351446>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5}
    cylinder { m*<-4.226493872134569,8.164965809277259,-2.2209066638373622>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5 }
    cylinder {  m*<-4.226493872134569,-8.164965809277259,-2.220906663837366>, <0.5825870371265556,-5.0117641047384395e-18,0.9737984987050873>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    