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
    sphere { m*<0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 1 }        
    sphere {  m*<0.3076449324588763,0.4630719171216834,2.8982714459105234>, 1 }
    sphere {  m*<2.801618221723444,0.43639581432773233,-1.3184928506612121>, 1 }
    sphere {  m*<-1.554705532175706,2.6628357833599594,-1.0632290906259976>, 1}
    sphere { m*<-2.4778357972840337,-4.4761149246049925,-1.563692881592985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3076449324588763,0.4630719171216834,2.8982714459105234>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5 }
    cylinder { m*<2.801618221723444,0.43639581432773233,-1.3184928506612121>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5}
    cylinder { m*<-1.554705532175706,2.6628357833599594,-1.0632290906259976>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5 }
    cylinder {  m*<-2.4778357972840337,-4.4761149246049925,-1.563692881592985>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5}

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
    sphere { m*<0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 1 }        
    sphere {  m*<0.3076449324588763,0.4630719171216834,2.8982714459105234>, 1 }
    sphere {  m*<2.801618221723444,0.43639581432773233,-1.3184928506612121>, 1 }
    sphere {  m*<-1.554705532175706,2.6628357833599594,-1.0632290906259976>, 1}
    sphere { m*<-2.4778357972840337,-4.4761149246049925,-1.563692881592985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3076449324588763,0.4630719171216834,2.8982714459105234>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5 }
    cylinder { m*<2.801618221723444,0.43639581432773233,-1.3184928506612121>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5}
    cylinder { m*<-1.554705532175706,2.6628357833599594,-1.0632290906259976>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5 }
    cylinder {  m*<-2.4778357972840337,-4.4761149246049925,-1.563692881592985>, <0.06690982771718473,0.33436183894135796,-0.08928332521002641>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    