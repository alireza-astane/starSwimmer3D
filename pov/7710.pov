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
    sphere { m*<-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 1 }        
    sphere {  m*<0.9529574438026053,0.45408422295463513,9.383830021426265>, 1 }
    sphere {  m*<8.320744642125408,0.16899197216237338,-5.186847407647672>, 1 }
    sphere {  m*<-6.575218551563592,6.692073345783011,-3.6960405044660645>, 1}
    sphere { m*<-3.7344776373664597,-7.65350334055454,-1.9789541031471218>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9529574438026053,0.45408422295463513,9.383830021426265>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5 }
    cylinder { m*<8.320744642125408,0.16899197216237338,-5.186847407647672>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5}
    cylinder { m*<-6.575218551563592,6.692073345783011,-3.6960405044660645>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5 }
    cylinder {  m*<-3.7344776373664597,-7.65350334055454,-1.9789541031471218>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5}

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
    sphere { m*<-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 1 }        
    sphere {  m*<0.9529574438026053,0.45408422295463513,9.383830021426265>, 1 }
    sphere {  m*<8.320744642125408,0.16899197216237338,-5.186847407647672>, 1 }
    sphere {  m*<-6.575218551563592,6.692073345783011,-3.6960405044660645>, 1}
    sphere { m*<-3.7344776373664597,-7.65350334055454,-1.9789541031471218>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9529574438026053,0.45408422295463513,9.383830021426265>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5 }
    cylinder { m*<8.320744642125408,0.16899197216237338,-5.186847407647672>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5}
    cylinder { m*<-6.575218551563592,6.692073345783011,-3.6960405044660645>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5 }
    cylinder {  m*<-3.7344776373664597,-7.65350334055454,-1.9789541031471218>, <-0.46621005039755703,-0.5358546909252824,-0.4654600756088869>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    