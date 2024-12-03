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
    sphere { m*<-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 1 }        
    sphere {  m*<-0.031367529788481646,0.27883682169699897,8.774781517080132>, 1 }
    sphere {  m*<6.720756058659383,0.10040259785858308,-5.387866439422484>, 1 }
    sphere {  m*<-3.104740406672917,2.1489282488576458,-1.9982846598950441>, 1}
    sphere { m*<-2.836953185635086,-2.7387636935462516,-1.8087383747324737>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.031367529788481646,0.27883682169699897,8.774781517080132>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5 }
    cylinder { m*<6.720756058659383,0.10040259785858308,-5.387866439422484>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5}
    cylinder { m*<-3.104740406672917,2.1489282488576458,-1.9982846598950441>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5 }
    cylinder {  m*<-2.836953185635086,-2.7387636935462516,-1.8087383747324737>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5}

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
    sphere { m*<-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 1 }        
    sphere {  m*<-0.031367529788481646,0.27883682169699897,8.774781517080132>, 1 }
    sphere {  m*<6.720756058659383,0.10040259785858308,-5.387866439422484>, 1 }
    sphere {  m*<-3.104740406672917,2.1489282488576458,-1.9982846598950441>, 1}
    sphere { m*<-2.836953185635086,-2.7387636935462516,-1.8087383747324737>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.031367529788481646,0.27883682169699897,8.774781517080132>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5 }
    cylinder { m*<6.720756058659383,0.10040259785858308,-5.387866439422484>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5}
    cylinder { m*<-3.104740406672917,2.1489282488576458,-1.9982846598950441>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5 }
    cylinder {  m*<-2.836953185635086,-2.7387636935462516,-1.8087383747324737>, <-1.4326467131513894,-0.18032437992768532,-1.1158601724124038>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    