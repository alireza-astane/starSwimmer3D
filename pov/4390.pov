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
    sphere { m*<-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 1 }        
    sphere {  m*<0.2704310622178809,0.1493693528038233,5.00038468165684>, 1 }
    sphere {  m*<2.54477369092311,0.005266755350022273,-1.9420254177672078>, 1 }
    sphere {  m*<-1.8115500629760373,2.2317067243822475,-1.6867616577319946>, 1}
    sphere { m*<-1.5437628419382055,-2.65598521802165,-1.4972153725694217>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2704310622178809,0.1493693528038233,5.00038468165684>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5 }
    cylinder { m*<2.54477369092311,0.005266755350022273,-1.9420254177672078>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5}
    cylinder { m*<-1.8115500629760373,2.2317067243822475,-1.6867616577319946>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5 }
    cylinder {  m*<-1.5437628419382055,-2.65598521802165,-1.4972153725694217>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5}

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
    sphere { m*<-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 1 }        
    sphere {  m*<0.2704310622178809,0.1493693528038233,5.00038468165684>, 1 }
    sphere {  m*<2.54477369092311,0.005266755350022273,-1.9420254177672078>, 1 }
    sphere {  m*<-1.8115500629760373,2.2317067243822475,-1.6867616577319946>, 1}
    sphere { m*<-1.5437628419382055,-2.65598521802165,-1.4972153725694217>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2704310622178809,0.1493693528038233,5.00038468165684>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5 }
    cylinder { m*<2.54477369092311,0.005266755350022273,-1.9420254177672078>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5}
    cylinder { m*<-1.8115500629760373,2.2317067243822475,-1.6867616577319946>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5 }
    cylinder {  m*<-1.5437628419382055,-2.65598521802165,-1.4972153725694217>, <-0.1899347030831472,-0.09676722003635194,-0.7128158923160263>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    