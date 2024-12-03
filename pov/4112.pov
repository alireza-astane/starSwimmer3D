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
    sphere { m*<-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 1 }        
    sphere {  m*<0.14412233720363998,0.08183783973727612,3.432876550861584>, 1 }
    sphere {  m*<2.5746218038513127,0.0212251796631127,-1.5716063536956>, 1 }
    sphere {  m*<-1.7817019500478342,2.2476651486953374,-1.3163425936603865>, 1}
    sphere { m*<-1.5139147290100023,-2.64002679370856,-1.1267963084978139>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14412233720363998,0.08183783973727612,3.432876550861584>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5 }
    cylinder { m*<2.5746218038513127,0.0212251796631127,-1.5716063536956>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5}
    cylinder { m*<-1.7817019500478342,2.2476651486953374,-1.3163425936603865>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5 }
    cylinder {  m*<-1.5139147290100023,-2.64002679370856,-1.1267963084978139>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5}

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
    sphere { m*<-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 1 }        
    sphere {  m*<0.14412233720363998,0.08183783973727612,3.432876550861584>, 1 }
    sphere {  m*<2.5746218038513127,0.0212251796631127,-1.5716063536956>, 1 }
    sphere {  m*<-1.7817019500478342,2.2476651486953374,-1.3163425936603865>, 1}
    sphere { m*<-1.5139147290100023,-2.64002679370856,-1.1267963084978139>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14412233720363998,0.08183783973727612,3.432876550861584>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5 }
    cylinder { m*<2.5746218038513127,0.0212251796631127,-1.5716063536956>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5}
    cylinder { m*<-1.7817019500478342,2.2476651486953374,-1.3163425936603865>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5 }
    cylinder {  m*<-1.5139147290100023,-2.64002679370856,-1.1267963084978139>, <-0.16008659015494406,-0.08080879572326141,-0.34239682824441675>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    