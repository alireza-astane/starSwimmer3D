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
    sphere { m*<-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 1 }        
    sphere {  m*<0.3485886558561694,0.1911566525761777,5.97033084727284>, 1 }
    sphere {  m*<2.52350768630308,-0.006103207166097496,-2.205939372623991>, 1 }
    sphere {  m*<-1.832816067596067,2.220336761866127,-1.950675612588778>, 1}
    sphere { m*<-1.5650288465582352,-2.66735518053777,-1.7611293274262054>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3485886558561694,0.1911566525761777,5.97033084727284>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5 }
    cylinder { m*<2.52350768630308,-0.006103207166097496,-2.205939372623991>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5}
    cylinder { m*<-1.832816067596067,2.220336761866127,-1.950675612588778>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5 }
    cylinder {  m*<-1.5650288465582352,-2.66735518053777,-1.7611293274262054>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5}

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
    sphere { m*<-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 1 }        
    sphere {  m*<0.3485886558561694,0.1911566525761777,5.97033084727284>, 1 }
    sphere {  m*<2.52350768630308,-0.006103207166097496,-2.205939372623991>, 1 }
    sphere {  m*<-1.832816067596067,2.220336761866127,-1.950675612588778>, 1}
    sphere { m*<-1.5650288465582352,-2.66735518053777,-1.7611293274262054>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3485886558561694,0.1911566525761777,5.97033084727284>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5 }
    cylinder { m*<2.52350768630308,-0.006103207166097496,-2.205939372623991>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5}
    cylinder { m*<-1.832816067596067,2.220336761866127,-1.950675612588778>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5 }
    cylinder {  m*<-1.5650288465582352,-2.66735518053777,-1.7611293274262054>, <-0.21120070770317712,-0.10813718255247172,-0.9767298471728105>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    