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
    sphere { m*<-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 1 }        
    sphere {  m*<0.2321449894533706,0.28436701629236427,8.545110594456492>, 1 }
    sphere {  m*<5.09106082887622,0.05083887672909046,-4.364678891012198>, 1 }
    sphere {  m*<-2.597468399367156,2.1652877512618294,-2.2926744245705963>, 1}
    sphere { m*<-2.3296811783293245,-2.722404191142068,-2.1031281394080255>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2321449894533706,0.28436701629236427,8.545110594456492>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5 }
    cylinder { m*<5.09106082887622,0.05083887672909046,-4.364678891012198>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5}
    cylinder { m*<-2.597468399367156,2.1652877512618294,-2.2926744245705963>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5 }
    cylinder {  m*<-2.3296811783293245,-2.722404191142068,-2.1031281394080255>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5}

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
    sphere { m*<-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 1 }        
    sphere {  m*<0.2321449894533706,0.28436701629236427,8.545110594456492>, 1 }
    sphere {  m*<5.09106082887622,0.05083887672909046,-4.364678891012198>, 1 }
    sphere {  m*<-2.597468399367156,2.1652877512618294,-2.2926744245705963>, 1}
    sphere { m*<-2.3296811783293245,-2.722404191142068,-2.1031281394080255>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2321449894533706,0.28436701629236427,8.545110594456492>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5 }
    cylinder { m*<5.09106082887622,0.05083887672909046,-4.364678891012198>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5}
    cylinder { m*<-2.597468399367156,2.1652877512618294,-2.2926744245705963>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5 }
    cylinder {  m*<-2.3296811783293245,-2.722404191142068,-2.1031281394080255>, <-0.9437644430655832,-0.1636219904085694,-1.3753770246921249>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    