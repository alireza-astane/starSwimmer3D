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
    sphere { m*<0.9646291631284423,0.5286739284705901,0.43622028290272696>, 1 }        
    sphere {  m*<1.2084062394642692,0.5721889334050847,3.425980867295717>, 1 }
    sphere {  m*<3.701653428526807,0.5721889334050845,-0.7913013411949017>, 1 }
    sphere {  m*<-2.721718882362444,6.273007784846883,-1.7433832977059975>, 1}
    sphere { m*<-3.8265825551597,-7.7787618098441085,-2.395980655514829>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2084062394642692,0.5721889334050847,3.425980867295717>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5 }
    cylinder { m*<3.701653428526807,0.5721889334050845,-0.7913013411949017>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5}
    cylinder { m*<-2.721718882362444,6.273007784846883,-1.7433832977059975>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5 }
    cylinder {  m*<-3.8265825551597,-7.7787618098441085,-2.395980655514829>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5}

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
    sphere { m*<0.9646291631284423,0.5286739284705901,0.43622028290272696>, 1 }        
    sphere {  m*<1.2084062394642692,0.5721889334050847,3.425980867295717>, 1 }
    sphere {  m*<3.701653428526807,0.5721889334050845,-0.7913013411949017>, 1 }
    sphere {  m*<-2.721718882362444,6.273007784846883,-1.7433832977059975>, 1}
    sphere { m*<-3.8265825551597,-7.7787618098441085,-2.395980655514829>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2084062394642692,0.5721889334050847,3.425980867295717>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5 }
    cylinder { m*<3.701653428526807,0.5721889334050845,-0.7913013411949017>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5}
    cylinder { m*<-2.721718882362444,6.273007784846883,-1.7433832977059975>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5 }
    cylinder {  m*<-3.8265825551597,-7.7787618098441085,-2.395980655514829>, <0.9646291631284423,0.5286739284705901,0.43622028290272696>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    