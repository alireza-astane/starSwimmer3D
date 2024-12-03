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
    sphere { m*<0.36066155547091705,0.8896573650355414,0.08091457211572331>, 1 }        
    sphere {  m*<0.601396660212609,1.0183674432158671,3.0684693432362766>, 1 }
    sphere {  m*<3.095369949477174,0.9916913404219161,-1.1482949533354603>, 1 }
    sphere {  m*<-1.2609538044219724,3.2181313094541437,-0.8930311933002463>, 1}
    sphere { m*<-3.5139945779322614,-6.4348245451983805,-2.164036740369491>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.601396660212609,1.0183674432158671,3.0684693432362766>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5 }
    cylinder { m*<3.095369949477174,0.9916913404219161,-1.1482949533354603>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5}
    cylinder { m*<-1.2609538044219724,3.2181313094541437,-0.8930311933002463>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5 }
    cylinder {  m*<-3.5139945779322614,-6.4348245451983805,-2.164036740369491>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5}

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
    sphere { m*<0.36066155547091705,0.8896573650355414,0.08091457211572331>, 1 }        
    sphere {  m*<0.601396660212609,1.0183674432158671,3.0684693432362766>, 1 }
    sphere {  m*<3.095369949477174,0.9916913404219161,-1.1482949533354603>, 1 }
    sphere {  m*<-1.2609538044219724,3.2181313094541437,-0.8930311933002463>, 1}
    sphere { m*<-3.5139945779322614,-6.4348245451983805,-2.164036740369491>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.601396660212609,1.0183674432158671,3.0684693432362766>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5 }
    cylinder { m*<3.095369949477174,0.9916913404219161,-1.1482949533354603>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5}
    cylinder { m*<-1.2609538044219724,3.2181313094541437,-0.8930311933002463>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5 }
    cylinder {  m*<-3.5139945779322614,-6.4348245451983805,-2.164036740369491>, <0.36066155547091705,0.8896573650355414,0.08091457211572331>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    